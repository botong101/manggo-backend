import threading

_lock = threading.Lock()

_status: dict = {
    'is_running':      False,
    'model_type':      None,
    'model_kind':      None,  # mobilenetv2 | hybrid_cnn
    'phase':           None,  # starting | downloading | preparing | training | evaluating | saving | done | error
    'progress':        0,
    'message':         '',
    'started_at':      None,
    'finished_at':     None,
    'output_filename': None,
    'accuracy':        None,
    'error':           None,
    'dataset_info':    None,
}


def get_status() -> dict:
    with _lock:
        return dict(_status)


def _set(**kwargs) -> None:
    with _lock:
        _status.update(kwargs)
