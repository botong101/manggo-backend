import threading

_lock = threading.Lock()

_status: dict = {
    'is_running':     False,
    'phase':          None,  # starting | scanning | extracting | saving | done | error
    'progress':       0,
    'message':        '',
    'started_at':     None,
    'finished_at':    None,
    'output_csv':     None,
    'rows_extracted': None,
    'error':          None,
}


def get_extraction_status() -> dict:
    with _lock:
        return dict(_status)


def _set(**kwargs) -> None:
    with _lock:
        _status.update(kwargs)


def try_start(**kwargs) -> bool:
    """Atomically check-and-set is_running. Returns True only if extraction was idle."""
    with _lock:
        if _status['is_running']:
            return False
        _status.update({'is_running': True, **kwargs})
        return True
