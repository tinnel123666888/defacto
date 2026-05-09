import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import socket
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed

from swift.rewards import ORM, orms
import logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def _collect_hosts_from_obj(obj: Any) -> List[str]:
    hosts: List[str] = []

    def ip_int_to_str(ip_num: int) -> Optional[str]:
        try:
            if ip_num < 0 or ip_num > 0xFFFFFFFF:
                return None
            return socket.inet_ntoa(struct.pack('!I', ip_num))
        except Exception:
            return None

    def walk(x: Any):
        if isinstance(x, dict):
            for key in ('ip', 'host', 'machineIp', 'serverIp', 'addr', 'address', 'head_node_ip'):
                val = x.get(key)
                if isinstance(val, str) and val.strip():
                    hosts.append(val.strip())
                elif isinstance(val, int) and key == 'ip':
                    ip_str = ip_int_to_str(val)
                    if ip_str:
                        hosts.append(ip_str)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    uniq = []
    seen = set()
    for host in hosts:
        if host not in seen:
            seen.add(host)
            uniq.append(host)
    return uniq


def _resolve_vllm_base_url(appid: str, broker_url: str, port: int = 8000) -> str:
    import requests

    resp = requests.post(broker_url, json={'appid': appid}, timeout=15)
    if not resp.ok:
        raise RuntimeError(f'broker status={resp.status_code}, body={resp.text[:200]}')
    data = resp.json()
    hosts = _collect_hosts_from_obj(data)
    if not hosts:
        raise RuntimeError('no host found in broker response')

    host = hosts[0]
    if host.startswith('http://') or host.startswith('https://'):
        host = host.split('://', 1)[1]
    host = host.split('/', 1)[0]
    host = host.split(':', 1)[0]
    return f'http://{host}:{port}/v1'


def _safe_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_tag(text: str, tag: str) -> str:
    text = _completion_to_text(text)
    m = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ''


def _completion_to_text(completion: Any) -> str:
    if completion is None:
        return ''
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        for k in ('content', 'text', 'completion', 'response', 'output'):
            v = completion.get(k)
            if isinstance(v, str):
                return v
            if isinstance(v, (dict, list, tuple)):
                t = _completion_to_text(v)
                if t:
                    return t
        return ''
    if isinstance(completion, (list, tuple)):
        parts: List[str] = []
        for x in completion:
            t = _completion_to_text(x)
            if t:
                parts.append(t)
        return '\n'.join(parts)
    try:
        return str(completion)
    except Exception:
        return ''


def _extract_prediction_text(completion: Any) -> str:
    answer = _extract_tag(completion, 'answer')
    # Enforce tagged final answer. If <answer> is missing, treat as empty.
    # This prevents rewarding answers hidden inside <think> or free-form text.
    return answer if answer else ''


def _extract_question_from_kwargs(idx: int, sol: Dict[str, Any], kwargs: Dict[str, Any]) -> str:
    for key in ('question', 'query', 'prompt', 'instruction'):
        val = sol.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    def _get_indexed(x: Any) -> Any:
        if isinstance(x, (list, tuple)) and idx < len(x):
            return x[idx]
        return x

    for key in ('questions', 'queries', 'prompts', 'prompt', 'query', 'messages', 'instruction'):
        if key not in kwargs:
            continue
        item = _get_indexed(kwargs.get(key))
        if isinstance(item, str) and item.strip():
            return item.strip()
        if isinstance(item, dict):
            for sub_key in ('question', 'query', 'prompt', 'content', 'text'):
                sub_val = item.get(sub_key)
                if isinstance(sub_val, str) and sub_val.strip():
                    return sub_val.strip()
            msgs = item.get('messages')
            if isinstance(msgs, list):
                for msg in reversed(msgs):
                    if isinstance(msg, dict) and msg.get('role') in {'user', 'human'}:
                        content = msg.get('content')
                        if isinstance(content, str) and content.strip():
                            return content.strip()
        if isinstance(item, list):
            for msg in reversed(item):
                if isinstance(msg, dict) and msg.get('role') in {'user', 'human'}:
                    content = msg.get('content')
                    if isinstance(content, str) and content.strip():
                        return content.strip()
    return ''


def _extract_bbox_list(completion: str) -> List[List[float]]:
    # Support <boxes>/<bbox> and compatibility alias <box>.
    bbox_text = _extract_tag(completion, 'boxes')
    if not bbox_text:
        bbox_text = _extract_tag(completion, 'bbox')
    if not bbox_text:
        bbox_text = _extract_tag(completion, 'box')
    if not bbox_text:
        return []
    if bbox_text.strip().lower() == 'null':
        return []
    obj = _safe_json_loads(bbox_text)
    if not isinstance(obj, list):
        return []
    out: List[List[float]] = []
    for box in obj:
        if isinstance(box, list) and len(box) == 4:
            try:
                out.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
            except Exception:
                continue
    return out


def _is_unknown_answer(answer: str, unknown_aliases: Sequence[str]) -> bool:
    if not answer:
        return False
    norm = answer.strip().lower()
    aliases = {a.strip().lower() for a in unknown_aliases if a}
    if norm in aliases:
        return True
    unknown_patterns = [
        r"\bcannot\s+determine\b",
        r"\bcan'?t\s+determine\b",
        r"\bcannot\s+tell\b",
        r"\bcan'?t\s+tell\b",
        r"\bnot\s+sure\b",
        r"\bunknown\b",
        r"\bunanswerable\b",
        r"\binsufficient\s+information\b",
        r"\bnot\s+enough\s+information\b",
        r"无法判断",
        r"无法确定",
        r"不确定",
        r"不知道",
        r"看不出来",
    ]
    return any(re.search(p, norm) for p in unknown_patterns)


def _normalize_text(text: str) -> str:
    text = (text or '').strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def _parse_binary_judge(content: str) -> Optional[float]:
    """Parse judge output robustly: accept plain `1`/`0` or text containing a standalone 1/0."""
    text = (content or '').strip()
    if not text:
        return None
    if text.startswith('1'):
        return 1.0
    if text.startswith('0'):
        return 0.0
    m = re.search(r'\b([01])\b', text)
    if m:
        return 1.0 if m.group(1) == '1' else 0.0
    return None


def _exact_match(pred: str, gt: str) -> bool:
    return _normalize_text(pred) == _normalize_text(gt)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax2 < ax1:
        ax1, ax2 = ax2, ax1
    if ay2 < ay1:
        ay1, ay2 = ay2, ay1
    if bx2 < bx1:
        bx1, bx2 = bx2, bx1
    if by2 < by1:
        by1, by2 = by2, by1

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class ViscotAnswerReward(ORM):
    """Answer accuracy reward with PARALLEL API calls"""

    def __init__(
        self,
        gamma_unk: float = 0.2,
        rho_unk: float = 1.0,
        gamma_guess: float = 0.6,
        gamma_corr: float = 0.9,
        max_workers: int = 64,  # 并行线程数 (增加默认值)
    ):
        self.gamma_unk = gamma_unk
        self.rho_unk = rho_unk
        self.gamma_guess = gamma_guess
        self.gamma_corr = gamma_corr
        self.max_workers = int(os.environ.get('VISCOT_API_MAX_WORKERS', str(max_workers)) or str(max_workers))
        self.api_timeout = float(os.environ.get('VISCOT_API_TIMEOUT', '10') or '10')
        self.api_max_retries = int(os.environ.get('VISCOT_API_MAX_RETRIES', '6') or '6')  # 增加默认重试次数
        self.api_retry_backoff = float(os.environ.get('VISCOT_API_RETRY_BACKOFF', '0.5') or '0.5')  # 重试延迟倍数
        self.profile_reward = os.environ.get('VISCOT_PROFILE_REWARD', '1').strip().lower() in {'1', 'true', 'yes', 'on'}
        self.profile_log_every = int(os.environ.get('VISCOT_PROFILE_LOG_EVERY', '10') or '10')
        self._call_counter = 0
        self._api_call_stats = {'total': 0, 'success': 0, 'failed': 0}
        self._api_call_stats_lock = threading.Lock()

        self.api_appid = 'Qwen3-235B-A22B-Instruct-2507'
        self.api_broker_url = 'http://astrastarlinkbroker.production.polaris:12534/OPEN_API_GetMachineListByApp'
        self.api_port = 8000
        self.api_key = 'EMPTY'
        self.api_base_url = os.environ.get('VISCOT_API_BASE_URL', '').strip()
        self.api_model = os.environ.get('VISCOT_API_MODEL', '').strip()
        self.unknown_aliases = [
            x.strip() for x in os.environ.get(
                'VISCOT_UNKNOWN_ALIASES',
                'Unknown,unanswerable,cannot determine,cannot tell,not sure,insufficient information,无法判断,无法确定,不确定,不知道,看不出来',
            ).split(',') if x.strip()
        ]

        self.request_log_every = int(os.environ.get('VISCOT_SAVE_REQUEST_EVERY', '10') or '10')
        default_log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'output',
            f'viscot_api_requests_every{self.request_log_every}.jsonl',
        )
        self.request_log_file = os.environ.get('VISCOT_SAVE_REQUEST_FILE', default_log_path).strip() or default_log_path
        self._request_counter = 0
        self._request_counter_lock = threading.Lock()

        self.reward_trace_every = int(os.environ.get('VISCOT_SAVE_REWARD_TRACE_EVERY', '20') or '20')
        default_trace_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'output',
            f'viscot_reward_trace_every{self.reward_trace_every}.jsonl',
        )
        self.reward_trace_file = os.environ.get('VISCOT_SAVE_REWARD_TRACE_FILE', default_trace_path).strip() or default_trace_path
        self._reward_trace_counter = 0
        self._reward_trace_counter_lock = threading.Lock()

        self._api_client = None
        self._api_disabled = False
        try:
            from openai import OpenAI

            if not self.api_base_url:
                self.api_base_url = _resolve_vllm_base_url(
                    appid=self.api_appid,
                    broker_url=self.api_broker_url,
                    port=self.api_port,
                )
            self._api_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url,
                max_retries=self.api_max_retries,
                timeout=self.api_timeout,
            )
            if not self.api_model:
                self.api_model = self._api_client.models.list().data[0].id
        except Exception:
            self._api_client = None
            self._api_disabled = True

    def _maybe_save_request(self, prompt: str, max_tokens: int) -> None:
        if self.request_log_every <= 0:
            return

        should_save = False
        req_idx = 0
        with self._request_counter_lock:
            self._request_counter += 1
            req_idx = self._request_counter
            should_save = (req_idx % self.request_log_every == 0)

        if not should_save:
            return

        try:
            save_obj = {
                'ts': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                'request_idx': req_idx,
                'base_url': self.api_base_url,
                'model': self.api_model,
                'request_body': {
                    'model': self.api_model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.0,
                    'max_tokens': max_tokens,
                },
            }
            os.makedirs(os.path.dirname(self.request_log_file), exist_ok=True)
            with open(self.request_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(save_obj, ensure_ascii=False) + '\n')
        except Exception:
            pass

    def _api_acc_single(self, pred: str, gt: str) -> Optional[float]:
        """Single API call with retry logic (thread-safe)"""
        if self._api_client is None or self._api_disabled:
            return None
        
        max_tokens = 24
        prompt = (
            'You are a strict answer judge. Compare Prediction and GroundTruth. '
            'If semantically equivalent output 1, else output 0. '
            'Output only one character: 1 or 0.\n'
            f'Prediction: {pred}\n'
            f'GroundTruth: {gt}\n'
        )
        self._maybe_save_request(prompt=prompt, max_tokens=max_tokens)
        
        # Retry loop with exponential backoff
        for retry_attempt in range(1 + self.api_max_retries):
            try:
                resp = self._api_client.chat.completions.create(
                    model=self.api_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                content = (resp.choices[0].message.content or '').strip()
                with self._api_call_stats_lock:
                    self._api_call_stats['total'] += 1
                    self._api_call_stats['success'] += 1
                parsed = _parse_binary_judge(content)
                if parsed is not None:
                    return parsed
                return None
            except Exception as e:
                with self._api_call_stats_lock:
                    self._api_call_stats['total'] += 1
                
                if retry_attempt < self.api_max_retries:
                    # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s, 16s
                    wait_time = self.api_retry_backoff * (2 ** retry_attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    with self._api_call_stats_lock:
                        self._api_call_stats['failed'] += 1
                    return None
        
        return None

    def _api_unknown_single(self, pred: str) -> Optional[float]:
        """Judge whether prediction semantically means unknown/unanswerable (with retry)."""
        if self._api_client is None or self._api_disabled:
            return None
        
        max_tokens = 8
        prompt = (
            'You are a strict unknownness judge. '
            'Determine whether Prediction semantically expresses unknown / cannot determine / not enough information. '
            'If yes output 1, else output 0. Output only one character: 1 or 0.\n'
            f'Prediction: {pred}\n'
        )
        self._maybe_save_request(prompt=prompt, max_tokens=max_tokens)
        
        # Retry loop with exponential backoff
        for retry_attempt in range(1 + self.api_max_retries):
            try:
                resp = self._api_client.chat.completions.create(
                    model=self.api_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                content = (resp.choices[0].message.content or '').strip()
                with self._api_call_stats_lock:
                    self._api_call_stats['total'] += 1
                    self._api_call_stats['success'] += 1
                parsed = _parse_binary_judge(content)
                if parsed is not None:
                    return parsed
                return None
            except Exception as e:
                with self._api_call_stats_lock:
                    self._api_call_stats['total'] += 1
                
                if retry_attempt < self.api_max_retries:
                    # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s, 16s
                    wait_time = self.api_retry_backoff * (2 ** retry_attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    with self._api_call_stats_lock:
                        self._api_call_stats['failed'] += 1
                    return None
        
        return None

    def _maybe_save_reward_trace(self, trace_obj: Dict[str, Any]) -> None:
        if self.reward_trace_every <= 0:
            return

        should_save = False
        with self._reward_trace_counter_lock:
            self._reward_trace_counter += 1
            trace_idx = self._reward_trace_counter
            should_save = (trace_idx % self.reward_trace_every == 0)

        if not should_save:
            return

        try:
            payload = {
                'ts': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                'trace_idx': trace_idx,
                **trace_obj,
            }
            os.makedirs(os.path.dirname(self.reward_trace_file), exist_ok=True)
            with open(self.reward_trace_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception:
            pass

    def _answer_acc(self, pred: str, gt: str) -> float:
        """Immediate exact match check (no API)"""
        if _exact_match(pred, gt):
            return 1.0
        return -1.0  # Sentinel: needs API call

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """PARALLEL API reward computation"""
        t0 = time.perf_counter()
        self._call_counter += 1

        # Phase 1: Extract all predictions and collect API tasks
        tasks = []
        for idx, (comp, sol) in enumerate(zip(completions, solution)):
            sol = sol or {}
            t = (sol.get('type') or sol.get('reward_case') or 'pos').lower()
            y_star = sol.get('y_star', '')
            pred_ans = _extract_prediction_text(comp)
            is_unk = 1.0 if _is_unknown_answer(pred_ans, self.unknown_aliases) else 0.0
            question = _extract_question_from_kwargs(idx, sol, kwargs)
            completion_text = _completion_to_text(comp)
            
            # Quick exact match
            acc = self._answer_acc(pred_ans, y_star)
            
            tasks.append({
                'idx': idx,
                'type': t,
                'pred_ans': pred_ans,
                'y_star': y_star,
                'is_unk': is_unk,
                'acc': acc,
                'question': question,
                'completion_text': completion_text,
            })

        # Phase 2: Batch API calls in parallel
        t1 = time.perf_counter()
        api_tasks = [
            (task['idx'], task['pred_ans'], task['y_star'])
            for task in tasks
            if task['acc'] < 0 and task['type'] in {'pos', 'rand'}
        ]  # pos/rand use answer-equivalence API

        cf_unknown_tasks = [(task['idx'], task['pred_ans']) for task in tasks
                            if task['type'] == 'cf' and (task['pred_ans'] or '').strip()]
        
        api_results = {}
        unk_api_results = {}
        if api_tasks and not self._api_disabled:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._api_acc_single, pred, gt): idx
                    for idx, pred, gt in api_tasks
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if result is not None:
                            api_results[idx] = result
                    except Exception:
                        pass  # Failed API call, default to 0.0

        if cf_unknown_tasks and not self._api_disabled:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._api_unknown_single, pred): idx
                    for idx, pred in cf_unknown_tasks
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if result is not None:
                            unk_api_results[idx] = result
                    except Exception:
                        pass
        t2 = time.perf_counter()

        # Phase 3: Compute final rewards
        rewards = []
        for task in tasks:
            idx = task['idx']
            t = task['type']
            acc = task['acc']
            is_unk = task['is_unk']
            unk_api_score = None
            is_unk_source = 'heuristic'
            
            # Use API result if available
            if t in {'pos', 'rand'} and acc < 0:
                acc = api_results.get(idx, 0.0)
                acc_source = 'api' if idx in api_results else 'api_fallback_0'
            elif t == 'cf' and acc < 0:
                acc = 0.0
                acc_source = 'cf_no_answer_api'
            else:
                acc_source = 'exact'
            
            # Compute reward
            if t in {'pos', 'rand'}:
                r = acc - self.gamma_unk * is_unk
                formula = {
                    'kind': 'pos_or_rand',
                    'acc': float(acc),
                    'gamma_unk': float(self.gamma_unk),
                    'is_unk': float(is_unk),
                    'reward': float(r),
                }
            else:  # cf
                if idx in unk_api_results:
                    unk_api_score = float(unk_api_results[idx])
                    prev_is_unk = is_unk
                    is_unk = max(is_unk, unk_api_score)
                    if prev_is_unk < 1.0 and is_unk >= 1.0:
                        is_unk_source = 'cf_unknown_api'
                    elif prev_is_unk >= 1.0:
                        is_unk_source = 'heuristic+api'
                guess_penalty = self.gamma_guess * (1.0 - is_unk)
                corr_penalty = self.gamma_corr * (1.0 if acc >= 1.0 else 0.0)
                r = self.rho_unk * is_unk - guess_penalty - corr_penalty
                formula = {
                    'kind': 'cf',
                    'rho_unk': float(self.rho_unk),
                    'is_unk': float(is_unk),
                    'gamma_guess': float(self.gamma_guess),
                    'guess_penalty': float(guess_penalty),
                    'gamma_corr': float(self.gamma_corr),
                    'corr_penalty': float(corr_penalty),
                    'acc': float(acc),
                    'unk_api_score': unk_api_score,
                    'is_unk_source': is_unk_source,
                    'reward': float(r),
                }

            self._maybe_save_reward_trace({
                'sample_idx_in_batch': idx,
                'type': t,
                'question': task.get('question', ''),
                'prediction': task.get('pred_ans', ''),
                'ground_truth': task.get('y_star', ''),
                'acc_source': acc_source,
                'is_unknown': float(is_unk),
                'formula': formula,
                'completion_preview': (task.get('completion_text', '') or '')[:1000],
                'timing': {
                    'phase_extract_s': float(t1 - t0),
                    'phase_api_s': float(t2 - t1),
                    'phase_reward_s': float(t3 - t2) if 't3' in locals() else None,
                    'total_s': float((t3 - t0) if 't3' in locals() else (time.perf_counter() - t0)),
                    'api_tasks': len(api_tasks),
                    'api_success': len(api_results),
                    'cf_unknown_api_tasks': len(cf_unknown_tasks),
                    'cf_unknown_api_success': len(unk_api_results),
                },
            })
            
            rewards.append(float(r))
        t3 = time.perf_counter()

        if self.profile_reward and self.profile_log_every > 0 and self._call_counter % self.profile_log_every == 0:
            with self._api_call_stats_lock:
                api_stats = dict(self._api_call_stats)
            api_success_rate = (api_stats['success'] / api_stats['total'] * 100) if api_stats['total'] > 0 else 0
            logger.warning(
                '[ViscotAnswerReward] call=%d batch=%d api_tasks=%d api_success=%d phase_extract=%.3fs phase_api=%.3fs phase_reward=%.3fs total=%.3fs max_workers=%d timeout=%.1fs retry=%d api_calls=(total=%d success=%d failed=%d rate=%.1f%%)',
                self._call_counter,
                len(tasks),
                len(api_tasks),
                len(api_results),
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t3 - t0,
                self.max_workers,
                self.api_timeout,
                self.api_max_retries,
                api_stats['total'],
                api_stats['success'],
                api_stats['failed'],
                api_success_rate,
            )
        
        return rewards


class ViscotFormatReward(ORM):
    """Format validity reward: R_fmt"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.profile_reward = os.environ.get('VISCOT_PROFILE_REWARD', '1').strip().lower() in {'1', 'true', 'yes', 'on'}
        self.profile_log_every = int(os.environ.get('VISCOT_PROFILE_LOG_EVERY', '10') or '10')
        self._call_counter = 0

        self.reward_trace_every = int(os.environ.get('VISCOT_SAVE_REWARD_TRACE_EVERY', '20') or '20')
        default_trace_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'output',
            f'viscot_reward_trace_every{self.reward_trace_every}.jsonl',
        )
        self.reward_trace_file = os.environ.get('VISCOT_SAVE_REWARD_TRACE_FILE', default_trace_path).strip() or default_trace_path
        self._reward_trace_counter = 0
        self._reward_trace_counter_lock = threading.Lock()

    def _maybe_save_reward_trace(self, trace_obj: Dict[str, Any]) -> None:
        if self.reward_trace_every <= 0:
            return

        should_save = False
        with self._reward_trace_counter_lock:
            self._reward_trace_counter += 1
            trace_idx = self._reward_trace_counter
            should_save = (trace_idx % self.reward_trace_every == 0)

        if not should_save:
            return

        try:
            payload = {
                'ts': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                'trace_idx': trace_idx,
                **trace_obj,
            }
            os.makedirs(os.path.dirname(self.reward_trace_file), exist_ok=True)
            with open(self.reward_trace_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception:
            pass

    @staticmethod
    def _format_detail(completion: str) -> Dict[str, Any]:
        text = _completion_to_text(completion)
        has_boxes = re.search(r'<boxes>.*?</boxes>', text, flags=re.DOTALL | re.IGNORECASE) is not None
        has_bbox = re.search(r'<bbox>.*?</bbox>', text, flags=re.DOTALL | re.IGNORECASE) is not None
        has_box = re.search(r'<box>.*?</box>', text, flags=re.DOTALL | re.IGNORECASE) is not None
        has_think = re.search(r'<think>.*?</think>', text, flags=re.DOTALL | re.IGNORECASE) is not None
        has_answer = re.search(r'<answer>.*?</answer>', text, flags=re.DOTALL | re.IGNORECASE) is not None
        # Expected format: <think><box>...</box>...reasoning...</think><answer>...</answer>
        # The <box>/<bbox>/<boxes> tag is INSIDE <think>, not before it.
        strict_pattern_boxes = r'<think>\s*<boxes>.*?</boxes>.*?</think>\s*<answer>.*?</answer>'
        strict_pattern_bbox = r'<think>\s*<bbox>.*?</bbox>.*?</think>\s*<answer>.*?</answer>'
        strict_pattern_box = r'<think>\s*<box>.*?</box>.*?</think>\s*<answer>.*?</answer>'
        strict_match = (
            re.search(strict_pattern_boxes, text, flags=re.DOTALL | re.IGNORECASE) is not None
            or re.search(strict_pattern_bbox, text, flags=re.DOTALL | re.IGNORECASE) is not None
            or re.search(strict_pattern_box, text, flags=re.DOTALL | re.IGNORECASE) is not None
        )
        bboxes = _extract_bbox_list(text)
        bbox_len_ok = all(len(b) == 4 for b in bboxes)
        is_valid = strict_match and bbox_len_ok
        return {
            'is_valid': is_valid,
            'has_boxes_tag': has_boxes,
            'has_bbox_tag': has_bbox,
            'has_box_tag': has_box,
            'has_think_tag': has_think,
            'has_answer_tag': has_answer,
            'strict_order_match': strict_match,
            'bbox_count': len(bboxes),
            'bbox_len_ok': bbox_len_ok,
        }

    @staticmethod
    def _is_format_valid(completion: str) -> bool:
        # Expected format: <think><box>...</box>...reasoning...</think><answer>...</answer>
        pattern_boxes = r'<think>\s*<boxes>.*?</boxes>.*?</think>\s*<answer>.*?</answer>'
        pattern_bbox = r'<think>\s*<bbox>.*?</bbox>.*?</think>\s*<answer>.*?</answer>'
        pattern_box = r'<think>\s*<box>.*?</box>.*?</think>\s*<answer>.*?</answer>'
        if (
            re.search(pattern_boxes, completion, flags=re.DOTALL | re.IGNORECASE) is None
            and re.search(pattern_bbox, completion, flags=re.DOTALL | re.IGNORECASE) is None
            and re.search(pattern_box, completion, flags=re.DOTALL | re.IGNORECASE) is None
        ):
            return False
        bboxes = _extract_bbox_list(completion)
        for b in bboxes:
            if len(b) != 4:
                return False
        return True

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        t0 = time.perf_counter()
        self._call_counter += 1
        rewards: List[float] = []
        for idx, comp in enumerate(completions):
            detail = self._format_detail(comp)
            r = self.alpha if detail['is_valid'] else 0.0
            rewards.append(float(r))

            sol = {}
            if isinstance(solution, (list, tuple)) and idx < len(solution):
                sol = solution[idx] or {}
            question = _extract_question_from_kwargs(idx, sol if isinstance(sol, dict) else {}, kwargs)
            self._maybe_save_reward_trace({
                'reward_component': 'format',
                'sample_idx_in_batch': idx,
                'type': (sol.get('type') or sol.get('reward_case') or 'pos').lower() if isinstance(sol, dict) else 'pos',
                'question': question,
                'format_detail': detail,
                'formula': {
                    'kind': 'format',
                    'alpha': float(self.alpha),
                    'reward': float(r),
                },
                'completion_preview': (_completion_to_text(comp) or '')[:1000],
                'timing': {
                    'total_s': float(time.perf_counter() - t0),
                },
            })

        t1 = time.perf_counter()
        if self.profile_reward and self.profile_log_every > 0 and self._call_counter % self.profile_log_every == 0:
            logger.warning(
                '[ViscotFormatReward] call=%d batch=%d total=%.3fs alpha=%.3f',
                self._call_counter,
                len(completions),
                t1 - t0,
                self.alpha,
            )
        return rewards


class ViscotSelectionReward(ORM):
    """Box selection accuracy reward: R_sel"""

    def __init__(
        self,
        beta_pos: float = 1.0,
        beta_neg: float = 1.0,
        gamma_empty: float = 0.5,
    ):
        self.beta_pos = beta_pos
        self.beta_neg = beta_neg
        self.gamma_empty = gamma_empty
        self.profile_reward = os.environ.get('VISCOT_PROFILE_REWARD', '1').strip().lower() in {'1', 'true', 'yes', 'on'}
        self.profile_log_every = int(os.environ.get('VISCOT_PROFILE_LOG_EVERY', '10') or '10')
        self._call_counter = 0

        self.reward_trace_every = int(os.environ.get('VISCOT_SAVE_REWARD_TRACE_EVERY', '20') or '20')
        default_trace_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'output',
            f'viscot_reward_trace_every{self.reward_trace_every}.jsonl',
        )
        self.reward_trace_file = os.environ.get('VISCOT_SAVE_REWARD_TRACE_FILE', default_trace_path).strip() or default_trace_path
        self._reward_trace_counter = 0
        self._reward_trace_counter_lock = threading.Lock()

    def _maybe_save_reward_trace(self, trace_obj: Dict[str, Any]) -> None:
        if self.reward_trace_every <= 0:
            return

        should_save = False
        with self._reward_trace_counter_lock:
            self._reward_trace_counter += 1
            trace_idx = self._reward_trace_counter
            should_save = (trace_idx % self.reward_trace_every == 0)

        if not should_save:
            return

        try:
            payload = {
                'ts': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                'trace_idx': trace_idx,
                **trace_obj,
            }
            os.makedirs(os.path.dirname(self.reward_trace_file), exist_ok=True)
            with open(self.reward_trace_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception:
            pass

    def _r_sel(self, t: str, pred_boxes: List[List[float]], r_plus: List[List[float]], r_minus: List[List[float]]) -> Tuple[float, Dict[str, Any]]:
        if t == 'cf':
            return 0.0, {
                'case': 'cf_no_selection_reward',
                'pred_box_count': len(pred_boxes),
                'r_plus_count': len(r_plus),
                'r_minus_count': len(r_minus),
                'avg_plus_iou': 0.0,
                'avg_minus_iou': 0.0,
            }
        if not pred_boxes:
            return -self.gamma_empty, {
                'case': 'empty_prediction',
                'pred_box_count': 0,
                'r_plus_count': len(r_plus),
                'r_minus_count': len(r_minus),
                'avg_plus_iou': 0.0,
                'avg_minus_iou': 0.0,
            }

        plus_scores = []
        minus_scores = []
        for b in pred_boxes:
            plus = max((_iou(b, r) for r in r_plus), default=0.0)
            minus = max((_iou(b, r) for r in r_minus), default=0.0)
            plus_scores.append(plus)
            minus_scores.append(minus)

        avg_plus = sum(plus_scores) / max(1, len(plus_scores))
        avg_minus = sum(minus_scores) / max(1, len(minus_scores))
        return self.beta_pos * avg_plus - self.beta_neg * avg_minus, {
            'case': 'normal',
            'pred_box_count': len(pred_boxes),
            'r_plus_count': len(r_plus),
            'r_minus_count': len(r_minus),
            'avg_plus_iou': float(avg_plus),
            'avg_minus_iou': float(avg_minus),
        }

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        t0 = time.perf_counter()
        self._call_counter += 1
        rewards: List[float] = []
        for idx, (comp, sol) in enumerate(zip(completions, solution)):
            sol = sol or {}
            t = (sol.get('type') or sol.get('reward_case') or 'pos').lower()
            r_plus = sol.get('r_plus_boxes') or []
            r_minus = sol.get('r_minus_boxes') or []
            pred_boxes = _extract_bbox_list(comp)
            
            r, sel_detail = self._r_sel(t=t, pred_boxes=pred_boxes, r_plus=r_plus, r_minus=r_minus)
            rewards.append(float(r))

            question = _extract_question_from_kwargs(idx, sol, kwargs)
            self._maybe_save_reward_trace({
                'reward_component': 'selection',
                'sample_idx_in_batch': idx,
                'type': t,
                'question': question,
                'prediction_boxes': pred_boxes,
                'r_plus_boxes': r_plus,
                'r_minus_boxes': r_minus,
                'selection_detail': sel_detail,
                'formula': {
                    'kind': 'selection',
                    'beta_pos': float(self.beta_pos),
                    'beta_neg': float(self.beta_neg),
                    'gamma_empty': float(self.gamma_empty),
                    'reward': float(r),
                },
                'completion_preview': (_completion_to_text(comp) or '')[:1000],
                'timing': {
                    'total_s': float(time.perf_counter() - t0),
                },
            })

        t1 = time.perf_counter()
        if self.profile_reward and self.profile_log_every > 0 and self._call_counter % self.profile_log_every == 0:
            logger.warning(
                '[ViscotSelectionReward] call=%d batch=%d total=%.3fs beta_pos=%.3f beta_neg=%.3f gamma_empty=%.3f',
                self._call_counter,
                len(rewards),
                t1 - t0,
                self.beta_pos,
                self.beta_neg,
                self.gamma_empty,
            )
        return rewards


orms['viscot_answer'] = ViscotAnswerReward
orms['viscot_format'] = ViscotFormatReward
orms['viscot_selection'] = ViscotSelectionReward
