from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt


def data_augmentation(image):
    # Generate image generator for data augmentation
    datagen = ImageDataGenerator(
        # preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    # load one image and reshape
    img = load_img(image)
    x = img_to_array(img) / 255.
    x = x.reshape((1,) + x.shape)

    # plot 10 augmented images of the loaded iamge
    plt.figure(figsize=(20, 10))
    plt.suptitle('Data Augmentation', fontsize=28)

    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.subplot(3, 5, i + 1)
        plt.grid(False)
        plt.imshow(batch.reshape(218, 178, 3))

        if i == 9:
            break
        i += 1
        plt.show()
