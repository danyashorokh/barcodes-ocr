
import argparse
import os

import cv2

from src.wrapper import BarcodeRecognizer


def main():
    parser = argparse.ArgumentParser(description='Demo script')
    parser.add_argument('-w', type=str, help='model weights path', dest='model_path', default='weights/model_ocr.zip')
    parser.add_argument('-i', type=str, help='input image path', dest='img_path')
    parser.add_argument('-d', type=str, help='device type', dest='device', default='cpu')
    parser.add_argument('--show', dest='show', action='store_true', default=False,
                        help='show image flag')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise ValueError(f'{args.model_path} doesn`t exist')

    if not os.path.exists(args.img_path):
        raise ValueError(f'{args.img_path} doesn`t exist')

    detector = BarcodeRecognizer(model_path=args.model_path, device=args.device)

    image = cv2.imread(args.img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preds = detector.predict(image)
    print(preds)

    if args.show:
        cv2.imshow('image', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
