"""
Entry point — delegates to train.py or evaluate.py.
  uv run python main.py train --model fno
  uv run python main.py train --model unet
  uv run python main.py eval  --model fno --checkpoint checkpoints/fno_best.pt
"""
import sys
import subprocess

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('train', 'eval'):
        print(__doc__)
        sys.exit(1)
    script = 'train.py' if sys.argv[1] == 'train' else 'evaluate.py'
    subprocess.run([sys.executable, script] + sys.argv[2:])
