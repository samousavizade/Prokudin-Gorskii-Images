import cv2 as cv
import matplotlib.pyplot as pp
import numpy as np
import tifffile as tif
import sys
import scipy.ndimage as nd


def search_template_recursive(image, template, delta, time=0):
    time += 1
    if time == 6 or image.shape[0] <= 50 or image.shape[1] <= 50:
        return min_ssd(image, template)

    else:
        # pyrDown image and template (1.Gaussain Blur 2.remove decussate rows and cols)
        next_image = cv.pyrDown(image)
        next_template = cv.pyrDown(template)

        # search optimum row, col recursively
        rows, cols = search_template_recursive(next_image, next_template, delta, time)

        rows *= 2
        cols *= 2

        # check
        rows = np.array([rows]) if type(rows) == np.int64 else rows
        cols = np.array([cols]) if type(cols) == np.int64 else cols

        template_height = template.shape[0]
        template_width = template.shape[1]
        image_height = image.shape[0]
        image_width = image.shape[1]

        argmin = (0, 0)
        minimum = sys.maxsize
        # brute force search optimum row, col for each possible choice in [rows-delta, rows+delta]*[cols-delta, cols+delta]
        for row, col in zip(rows, cols):
            row_low = row - delta
            row_high = row + delta + template_height
            col_low = col - delta
            col_high = col + delta + template_width

            if row_low < 0:
                row_low = 0
            if col_low < 0:
                col_low = 0
            if row_high >= image_height:
                row_high = image_height
            if col_high >= image_width:
                col_high = image_width

            r, c, value = min_ssd(
                image[row_low:row_high, col_low:col_high],
                template,
                return_value=True
            )

            # set argmin and min
            if minimum > value:
                minimum = value
                argmin = (r + row_low, c + col_low)

    # return argmin place
    return argmin


def search_template(image, template, delta):
    result_row, result_col = search_template_recursive(image, template, delta=delta)
    result_row = result_row[0]
    result_col = result_col[0]
    print('optimum_row:', result_row - offset)
    print('optimum_col:', result_col - offset)
    return result_row, result_col


def min_ssd(image, template, return_value=False):
    ssd = np.zeros((image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1), dtype='int')
    ssd_h = ssd.shape[0]
    ssd_w = ssd.shape[1]
    template_h = template.shape[0]
    template_w = template.shape[1]

    # brute force search optimum row, col per for each possible choice
    for row in range(ssd_h):
        for col in range(ssd_w):
            ssd[row, col] = ((image[row:row + template_h, col:col + template_w] - template) ** 2).sum()

    # get rows, cols number
    rows = np.array(np.where(ssd == ssd.min())[0])
    cols = np.array(np.where(ssd == ssd.min())[1])

    if return_value:
        return rows, cols, ssd.min()

    return rows, cols


offset = 125


def main():
    # read .tif image
    input: np.ndarray = tif.imread('melons.tif')

    m = input.max()
    # change type and intensity scale
    if m >= 2 ** 8:
        max_intensity = 2 ** 16
        dtype = 'uint8'
        input = (input / 256).astype(dtype)
    else:
        max_intensity = 2 ** 8
        dtype = 'uint8'
        input = input.astype(dtype)

    # separate image to 3 image with equal height
    input_height = input.shape[0]
    mod = input_height % 3
    if mod != 0:
        input = input[:-mod, :]

    input_height = input.shape[0]
    melon = np.zeros((input_height // 3, input.shape[1], 3), dtype=dtype)

    # create melon rgb numpy array from channels
    melon[:, :, 0] = input[2 * input_height // 3:3 * input_height // 3, :]
    melon[:, :, 1] = input[1 * input_height // 3:2 * input_height // 3, :]
    melon[:, :, 2] = input[0 * input_height // 3:1 * input_height // 3, :]

    # offset for blue and red channel
    red = melon[:, :, 0]
    red = red[offset:-offset, offset:-offset]
    blue = melon[:, :, 2]
    blue = blue[offset:-offset, offset:-offset]
    green = melon[:, :, 1]

    # create empty output matrix
    output = np.zeros((green.shape[0], green.shape[1], 3), dtype=dtype)

    # LoG / laplacian of gaussian on r g b channel and get edges
    red_edges = nd.gaussian_laplace(red, sigma=1.5)
    green_edges = nd.gaussian_laplace(green, sigma=1.5)
    blue_edges = nd.gaussian_laplace(blue, sigma=1.5)

    # put green channel on output
    output[:, :, 1] += green

    # search optimum row, col for image=green_edges, template=blue_edges
    image = green_edges
    template = blue_edges
    print('Blue Channel as Template On Green Channel:')
    optimum_row, optimum_col = search_template(image, template, delta=5)
    print('\n')
    output[optimum_row: optimum_row + blue.shape[0], optimum_col:optimum_col + blue.shape[1], 2] += blue

    # search optimum row, col for image=green_edges, template=red_edges
    image = green_edges
    template = red_edges
    print('Red Channel as Template On Green Channel:')
    optimum_row, optimum_col = search_template(image, template, delta=5)
    print('\n')
    output[optimum_row: optimum_row + red.shape[0], optimum_col:optimum_col + red.shape[1], 0] += red

    # remove colored border
    output[output[:, :, 0] == 0] = 0
    output[output[:, :, 1] == 0] = 0
    output[output[:, :, 2] == 0] = 0

    print('result saved')
    pp.imsave('res04.jpg', output)

    # plot edges and source channels
    # fig, axs = pp.subplots(nrows=2, ncols=3)
    # fig: pp.Figure
    # ax1: pp.Axes
    # ax2: pp.Axes
    # ax3: pp.Axes
    # fig.set_size_inches(14, 6)

    # try:
    #     axs[0, 0].imshow(red, cmap='gray', vmin=0, vmax=max_intensity)
    #     axs[1, 0].imshow(red_edges, cmap='gray', vmin=0, vmax=max_intensity)
    #     axs[0, 1].imshow(green, cmap='gray', vmin=0, vmax=max_intensity)
    #     axs[1, 1].imshow(green_edges, cmap='gray', vmin=0, vmax=max_intensity)
    #     axs[0, 2].imshow(blue, cmap='gray', vmin=0, vmax=max_intensity)
    #     axs[1, 2].imshow(blue_edges, cmap='gray', vmin=0, vmax=max_intensity)
    #     while not fig.waitforbuttonpress(): pass
    # except:
    #     print('Result Plotted')
    #     pass

    # while not fig.waitforbuttonpress(): pass


if __name__ == '__main__':
    main()
