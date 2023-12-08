import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure


def plot_digit_images(df, keep_plot_titles=False):
    # images of digits
    digit_per_class = df.groupby('label').first()

    # 1 digit gray
    fig = plt.figure(figsize=(8, 6))
    digit = digit_per_class.loc[0]
    plt.imshow(digit.values.reshape(28, 28), cmap='gray')
    if keep_plot_titles:
        plt.title("Sample Image of a Digit")
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    # plt.show()
    fig.savefig('./plots/digit_single_gray.png')

    # 1 digit
    fig = plt.figure(figsize=(8, 6))
    digit = digit_per_class.loc[0]
    plt.imshow(digit.values.reshape(28, 28))
    if keep_plot_titles:
        plt.title("Sample Image of a Digit")
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    # plt.show()
    fig.savefig('./plots/digit_single.png')

    # 1 digit for every class gray
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        digit = digit_per_class.loc[i]
        ax.imshow(digit.values.reshape(28, 28), cmap='gray')
        ax.set_title(f"Digit {i}")
        ax.axis('off')
    if keep_plot_titles:
        plt.title("Sample Images of All Digits")
    #plt.show()
    fig.savefig('./plots/digit_all_gray.png')

    # 1 digit for every class
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        digit = digit_per_class.loc[i]
        ax.imshow(digit.values.reshape(28, 28))
        ax.set_title(f"Digit {i}")
        ax.axis('off')
    if keep_plot_titles:
        plt.title("Sample Images of All Digits")
    # plt.show()
    fig.savefig('./plots/digit_all.png')


def plot_dataset_statistics(df, keep_plot_titles=False, print_stats=False):
    # barplot of class distribution with proportions
    class_counts = df.value_counts(subset='label').sort_index()
    class_counts_norm = df.value_counts(subset='label', normalize=True).sort_index()
    if print_stats:
        print(pd.concat([class_counts, class_counts_norm], axis=1))
        print(df.describe())

    fig = plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar', rot=0)
    if keep_plot_titles:
        plt.title('Class Distribution with Proportions')
    plt.xlabel('Digit Label')
    plt.ylabel('Sample Count')
    for i, count in enumerate(class_counts):
        plt.text(i, count + 15, f"{class_counts_norm[i]*100:.2f}%", ha='center')  # proportions
    #plt.show()
    fig.savefig('./plots/stat_bar_class_distribution.png')

    # histogram of pixel value frequencies
    pixel_values = df.iloc[:, 1:].values.flatten()

    fig = plt.figure(figsize=(8, 6))
    plt.hist(pixel_values, bins=50, log=True)
    if keep_plot_titles:
        plt.title('Pixel Value Frequencies')
    plt.xlabel('Pixel Value')
    plt.ylabel('Logarithmic Frequency')
    #plt.show()
    fig.savefig('./plots/stat_hist_pixel_value_frequency.png')

    # heatmap for mean of pixel values across for separate classes
    mean_digits = df.groupby('label').mean()  # by class
    mean_digits = mean_digits.values.reshape(10, 28, 28)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns for 10 digits
    for digit, ax in enumerate(axes.flatten()):
        im = ax.imshow(mean_digits[digit], cmap='hot', interpolation='nearest')
        ax.set_title(f"Digit {digit}")
        ax.axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    if keep_plot_titles:
        plt.title('Mean of Pixel Values per Digit')
    #plt.show()
    fig.savefig('./plots/stat_heatmap_pixel_mean_value_classes.png')

    # heatmap for mean of pixel values across all classes
    mean_pixels = np.mean(df.iloc[:, 1:], axis=0)  # mean for every pixel
    mean_pixels = mean_pixels.values.reshape(28, 28)  # Reshape to 28x28 grid

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(mean_pixels, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    if keep_plot_titles:
        plt.title('Mean of Pixel Values')
    #plt.show()
    fig.savefig('./plots/stat_heatmap_pixel_mean_value_overall.png')

    # heatmap for standard deviation of pixel values across for separate classes
    std_digits = df.groupby('label').std()  # by class
    std_digits = std_digits.values.reshape(10, 28, 28)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns for 10 digits
    for digit, ax in enumerate(axes.flatten()):
        im = ax.imshow(std_digits[digit], cmap='hot', interpolation='nearest')
        ax.set_title(f"Digit {digit}")
        ax.axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    if keep_plot_titles:
        plt.title('Standard Deviation of Pixel Values per Digit')
    #plt.show()
    fig.savefig('./plots/stat_heatmap_pixel_std_value_classes.png')

    # heatmap for standard deviation of pixel values across all classes
    std_pixels = np.std(df.iloc[:, 1:], axis=0)  # std for every pixel
    std_pixels = std_pixels.values.reshape(28, 28)  # Reshape to 28x28 grid

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(std_pixels, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    if keep_plot_titles:
        plt.title('Standard Deviation of Pixel Values')
    #plt.show()
    fig.savefig('./plots/stat_heatmap_pixel_std_value_overall.png')

    # combine overall mean and std plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(mean_pixels, cmap='hot', interpolation='nearest')
    axes[0].axis('off')
    axes[0].set_title('Mean of Pixel Values')
    axes[1].imshow(std_pixels, cmap='hot', interpolation='nearest')
    axes[1].axis('off')
    axes[1].set_title('Standard Deviation of Pixel Values')
    fig.colorbar(im, ax=axes.ravel().tolist())
    #plt.show()
    fig.savefig('./plots/stat_heatmap_pixel_mean_std_value_overall.png')


def plot_ink_feature(digits, keep_plot_titles=False, print_stats=False):
    # sum of ink on every row
    ink_feature = np.array([sum(row) for row in digits])

    ink_mean = [np.mean(ink_feature[labels == i]) for i in range(10)]  # mean for each digit class
    ink_std = [np.std(ink_feature[labels == i]) for i in range(10)]  # std for each digit clas
    ink_stats = pd.concat([pd.Series(ink_mean, name="mean"), pd.Series(ink_std, name="std")], axis=1)
    if print_stats:
        print(ink_stats)

    # error bar with mean and standard deviation of ink feature
    fig = plt.figure(figsize=(8, 6))
    plt.errorbar(ink_stats.index, ink_stats['mean'], yerr=ink_stats['std'], fmt='o')
    plt.xlabel('Digit Label')
    plt.ylabel('Mean Ink')
    if keep_plot_titles:
        plt.title('Mean Ink with Standard Deviation by Digit Label')
    plt.xticks(range(10))
    #plt.show()
    fig.savefig('./plots/ink.png')


def plot_hog_feature(digits, digit_idx=30):
    # plot showing histogram of oriented gradients
    image = digits[digit_idx]
    image = np.reshape(image, (28, 28))
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1), feature_vector=True,
                        visualize=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input Image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    #plt.show()
    plt.savefig('./plots/hog.png')


def calculate_ink_feature(digits):
    # sum of ink on every row
    ink_feature = np.array([sum(row) for row in digits])

    return scale(ink_feature).reshape(-1, 1)


def calculate_hog_feature(digits):
    # histogram of oriented gradients
    return np.array([hog(np.reshape(row, (28, 28)),
                         orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), feature_vector=True)
                     for row in digits])


def calculate_extra_features():
    # row color standard deviation
    color_std = np.array([np.std(row) for row in digits])
    color_std = scale(color_std).reshape(-1, 1)

    # length (norm)
    img_length = np.array([np.sqrt(np.sum([i**2 for i in row])) for row in digits])
    img_length = scale(img_length).reshape(-1,1)

    # max spread of digit width as a feature
    spread = []

    for idx, row in enumerate(digits):
        start = 0
        end = 28
        total = 784

        global_max_idx = 0
        global_min_idx = 0
        while end <= total:
            chunk = row[start:end]
            nonzero = np.nonzero(chunk)
            max_idx = nonzero[-1] if len(nonzero) > 0 else 0
            min_idx = nonzero[0] if len(nonzero) > 0 else 28

            if max_idx > global_max_idx and chunk.any() != 0:
                global_max_idx = max_idx

            if global_min_idx == 0 and chunk.any() != 0:
                global_min_idx = min_idx
            start += 28
            end += 28
        spread.append(global_max_idx - global_min_idx)

    index_spread = np.asarray(spread)
    index_spread = scale(index_spread).reshape(-1,1)

    return color_std, img_length, index_spread


def ink_model(labels, ink_feature, keep_plot_titles=False):
    classifier = LogisticRegression()
    classifier.fit(ink_feature, labels)
    predictions = classifier.predict(ink_feature)

    # plot confusion matrix and show results
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    if keep_plot_titles:
        plt.title("Confusion Matrix for Classifier with only Ink Feature")
    #plt.show()
    plt.savefig('./plots/confusion_matrix_ink.png')

    class_report = classification_report(labels, predictions)
    print(class_report)

    return classifier


def ink_hog_model(labels, ink_feature, hog_feature, keep_plot_titles=False):
    # hog + ink model
    features = np.column_stack([ink_feature, hog_feature])

    classifier = LogisticRegression()
    classifier.fit(features, labels)
    predictions = classifier.predict(features)

    # plot confusion matrix and show results
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    if keep_plot_titles:
        plt.title("Confusion Matrix for Classifier with only Ink Feature")
    #plt.show()
    plt.savefig('./plots/confusion_matrix_ink_hog.png')

    class_report = classification_report(labels, predictions)
    print(class_report)

    return classifier


mnist_data = pd.read_csv('mnist.csv')

# exploratory analysis
plot_digit_images(mnist_data)
plot_dataset_statistics(mnist_data)

mnist_data = mnist_data.values
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]

# ink feature only model
ink_feature = calculate_ink_feature(digits)
plot_ink_feature(digits)
trained_ink_model = ink_model(labels, ink_feature)

# ink and hog feature model
hog_feature = calculate_hog_feature(digits)
plot_hog_feature(digits)
trained_ink_hog_model = ink_hog_model(labels, ink_feature, hog_feature)
