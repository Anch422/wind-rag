"""Isolated benchmark entry point; terminating it closes active HTTP sockets."""
import json
import sys
import joblib


def send(event, data):
    print(json.dumps({'event': event, 'data': data}, ensure_ascii=True), flush=True)


def main(path):
    from .experiments import sequential_runs
    payload = joblib.load(path)
    try:
        folders = sequential_runs(payload['engine'], payload['corpus_hash'], payload['questions'],
                                  payload['models'], payload['methods'], payload['directory'],
                                  progress=lambda message: send('progress', message),
                                  completed=lambda folder: send('completed', folder), staging=payload['staging'])
        send('done', folders)
    except Exception as exc:
        send('error', f'{type(exc).__name__}: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
