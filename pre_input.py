import os

dir = '/home/uc3m1/PycharmProjects/siameseFaceNet/data/datasets/dataset_prueba'


def generate_txt_with_all_images(path):
    with open(path, 'w') as out:
        for person in os.listdir(dir):
            path = os.path.join(dir, person)

            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                out.write(img_path + '\n')
                print(img_path)


def main():
    generate_txt_with_all_images("./data/all_img_paths.txt")


if __name__ == '__main__':
    main()
