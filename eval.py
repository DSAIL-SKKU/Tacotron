import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer

sentences = [
  # 장기하와 얼굴들 ㅋ 가사:
  '너는 쿨쿨 자나봐.',
  '정성스럽게 적었던 거야.',
  '이모티콘 하나마저 조심스럽게 정했어.',

  '모든 걸 마무리해버렸어.',
  '나는 큰 결심을 하고서 보낸 문잔데.',
  # 장기하와 얼굴들 새해복 가사:
  '완전히 쾅 닫힌 대화창뿐이네.',
  '새해복 많이 많으세요.',
  '새해 복만으로는 안돼.',
  '너가 잘해야지',
]


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args):
    print(hparams_debug_string())
    synth = Synthesizer()
    synth.load(args.checkpoint)
    base_path = get_output_base_path(args.checkpoint)
    for i, text in enumerate(sentences):
        path = '%s-%d.wav' % (base_path, i)
        print('Synthesizing: %s' % path)
        with open(path, 'wb') as f:
            f.write(synth.synthesize(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
    main()
