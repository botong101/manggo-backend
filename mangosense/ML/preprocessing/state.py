import threading

_lock = threading.Lock()

_status = {
    'is_running':  False,
    'model_type':  None,
    'phase':       None,
    'progress':    0,
    'message':     '',
    'started_at':  None,
    'finished_at': None,
    'processed':   None,
    'failed':      None,
    'error':       None,
}


def get_status() -> dict:
    with _lock:
        return dict(_status)


def _set(**kwargs) -> None:
    with _lock:
        _status.update(kwargs)


def try_start(**kwargs) -> bool:
    with _lock:
        if _status['is_running']:
            return False
        _status.update({'is_running': True, **kwargs})
        return True
