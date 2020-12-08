import sys
import numpy as np
from scipy.spatial.distance import pdist
from scipy import ndimage as ndi
from skimage import (
    morphology,
    measure,
    io as skio,
)
import matplotlib.pyplot as plt
import cv2

sam = cv2.imread("data/sample2.jpg")
h,w,c = sam.shape
sam2 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        sam2[i][j] = sam[i][j][0]

for i in range(h):
    for j in range(w):
        if sam2[i][j] > 80:
            sam2[i][j] = 255

t_1 = sam2[:,:60]
t_2 = sam2[:,60:120]
t_3 = sam2[:,120:180]


# cv2.imwrite("sam2.jpg",sam2) 
cv2.imwrite("data/t_1.jpg",t_1)
cv2.imwrite("data/t_2.jpg",t_2)
cv2.imwrite("data/t_3.jpg",t_3) 

def find_contour(image):
    """
    Manually trace contour

    NOTE: Currently not used, but included as reference / starting
    point should we need a better set of contour pixels
    """
    contour = image ^ ndi.binary_erosion(image)
    points = np.array([(x, y) for x, y in zip(*contour.nonzero())])
    return points

def cross_ratio_spectrum(image, debug=False):
    convex_hull = morphology.convex_hull_image(image)

    border_pixels = measure.find_contours(
        ndi.binary_erosion(convex_hull, iterations=2), 0.5
    )
    assert len(border_pixels) == 1, "Not 1 contour found"
    border_pixels = border_pixels[0]
    border_pixels = border_pixels[:-1]
    # Debugging
    if debug:
        plt.figure("Input image and border pixels")
        plt.imshow(image, cmap="gray")
        plt.plot(border_pixels[:, 1], border_pixels[:, 0], "-b")
        plt.plot(border_pixels[:, 1], border_pixels[:, 0], ".r")
    p1 = border_pixels[0]
    # function
    spectrum = []
    for index, pk in enumerate(border_pixels[1:], start=1):
        spectrum.append(
            get_cross_ratio(
                p1, 
                pk, 
                image, 
                debug="" if not debug else f"{index:06d}"))
    return spectrum

def profile_line(image, src, dst):
    order = 0
    mode = "constant"
    linewidth = 1
    cval = -1

    perp_lines = measure.profile._line_profile_coordinates(
        src, dst, linewidth=linewidth
    )
    if image.ndim == 3:
        pixels = [
            ndi.map_coordinates(
                image[..., i],
                perp_lines,
                prefilter=order > 1,
                order=order,
                mode=mode,
                cval=cval,
            )
            for i in range(image.shape[2])
        ]
        pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
    else:
        pixels = ndi.map_coordinates(
            image, perp_lines, prefilter=order > 1, order=order, mode=mode, cval=cval
        )
    pixels = np.flip(pixels, axis=1)

    return pixels, perp_lines

def get_cross_ratio(p1, pk, image, debug=False):

    # To get the cross-ratio for p1-pk, we need the intersections
    intersections = get_intersections(p1, pk, image, debug=debug)
    # As written in the manuscript, handle 0 or 1 intersections here
    if len(intersections) == 0:
        return -1
    if len(intersections) == 1:
        return 0
    # Otherwise, calculate the cross-ratio using the
    # first two detected intersections
    return cross_ratio(p1, intersections[0], intersections[1], pk)   

def get_intersections(p1, pk, image, debug=False):
    """
    Get the intersections from p1-->pk (~i.e. steps in the image)
    """
    # We use a very slightly modified version of skimage.measure.profile_line
    # included in this source, in order to get both the line profile and the
    # points in the line profile, from p1 to pk
    profile, points = profile_line(image, p1, pk)
    # Profile comes back as an annoying 2d thing because it's
    # Also intended for width>1 lines
    # We set linewidth to 1 inside profile_line, so we can
    # safely "flatten" the profile
    profile = profile.flatten()
    # Similarly points comes back 2 x n-points x 1... for the same reason
    # So we remove the unnecessary last dimension using squeeze, and
    # transpose the array so it's n-points x 2
    points = points.squeeze().T
    # Now the intersections are simply points where the profile steps
    # up or down

    # NOTE: This is where the code can be made more robust to try and remove 
    # the spikes in the spectrum
    steps = np.diff(profile)
    locations = steps.nonzero()[0]
    intersections = []
    if len(locations) > 0:
        intersections = points[locations]
    if debug:
        plt.figure(f"get_intersections: {p1} -> {pk}")
        plt.subplot(121)
        plt.imshow(image, cmap="gray")
        plt.plot(points[:, 1], points[:, 0], ".r")
        if len(intersections) > 0:
            plt.plot(intersections[:, 1], intersections[:, 0], "ob", mfc="none")
        plt.legend("Profile locations")
        plt.subplot(122)
        plt.plot(profile)
        plt.savefig(f"debug_intersections_{id(image)}_{debug}_from_{p1}_to_{pk}.png")
        plt.close()


    return intersections

def our_dist(p1, p2):
    return np.sqrt(np.sum((np.array(p2)-np.array(p1))**2))

def cross_ratio(p1, p2, p3, p4):
    """
    Returns the cross ratio of the four input points

    Args:
    Returns:
        The cross ratio
    """
    # Relatively simple cross-ratio formula
    # based on point-point distances
    d13 = pdist((p1, p3))
    d23 = pdist((p2, p3))
    d14 = pdist((p1, p4))
    d24 = pdist((p2, p4))
    return (d13 / d23) / (d14 / d24)

def recognition_process(q_image, t_image, debug=False):
    """
    Q - Query character, with pixel sequence on convex hull {Q1,...,Qn}
    T - Template character with pixel sequence on convex hull {T1,...,Tm}
    1. Compare pair-wise similarity spectra of T and Q
    2. Find pixel-level correspondence and estimate similarity between T and Q
    """
    spectrum_q = cross_ratio_spectrum(q_image, debug=debug)
    spectrum_t = cross_ratio_spectrum(t_image, debug=debug)
    plt.figure("Spectra")
    plt.plot(spectrum_q, "-b", label="Q")
    plt.plot(spectrum_t, "-r", label="T")
    plt.legend()
    plt.savefig(f"spectra_{id(q_image)}_{id(t_image)}.png")
    plt.close()

    alignment = dtw(spectrum_q, spectrum_t)
    print(alignment)

    # To show the alignment
    dtw(
        spectrum_q,
        spectrum_t,
        keep_internals=True,
        step_pattern=rabinerJuangStepPattern(6, "c"),
    ).plot(type="twoway", offset=-2)

    
def load_im_as_binary(filename):
    """
    We need to load the letters as binary images
    so we provide a light wrapper around skimage.io.imread
    """
    image = skio.imread(filename)
    # If colour, assume we can just work with red channel
    if image.ndim == 3:
        image = image[:, :, 0]
    # Turn into binary images; note we invert - want letters to be 1
    image = image < image.max() / 2
    return image
                     
def main():

    template_im = load_im_as_binary("data/A.png")
    template_im_test = load_im_as_binary("data/B.png")
    C = load_im_as_binary("data/C.png")
    D = load_im_as_binary("data/D.png") 
    E = load_im_as_binary("data/E.png") 
    F = load_im_as_binary("data/F.png") 
    G = load_im_as_binary("data/G.png")
    H = load_im_as_binary("data/H.png") 
    I = load_im_as_binary("data/I.png") 
    J = load_im_as_binary("data/J.png") 
    K = load_im_as_binary("data/K.png") 
    L = load_im_as_binary("data/L.png") 
    M = load_im_as_binary("data/M.png")
    N = load_im_as_binary("data/N.png") 
    O = load_im_as_binary("data/O.png") 
    P = load_im_as_binary("data/P.png")   
    G_1 = load_im_as_binary("data/t_1.jpg")
    G_2 = load_im_as_binary("data/t_2.jpg")
    G_3 = load_im_as_binary("data/t_3.jpg")

    # binarization
    plt.figure()
    plt.imshow(template_im, cmap="gray")
    plt.title("binary image")
    plt.gcf().set_facecolor("white")

    # convex hull
    convex_hull = morphology.convex_hull_image(template_im)
  
    plt.figure()
    plt.imshow(convex_hull, cmap="gray")
    plt.title('Convex hull image')
    plt.gcf().set_facecolor("white")

    # erosion

    erosion_image = ndi.binary_erosion(convex_hull, iterations=2)

    plt.figure()
    plt.imshow(erosion_image)
    plt.title('Erosion image')
    plt.gcf().set_facecolor("white")

    # contours

    border_pixels = measure.find_contours(
        erosion_image, 0.5
    )
    
    # As find_contours trys to find contours of all enclosed shapes,
    # it returns a list - one per enclosed shape
    # We expect just one shape (for now)

    # print(len(border_pixels))
    assert len(border_pixels) == 1, "Not 1 contour found"
    border_pixels = border_pixels[0]
    # Remove the final border pixel as find_contours adds the initial
    # point at the end
    border_pixels = border_pixels[:-1]
   
    plt.figure("Input image and border pixels")
    plt.imshow(template_im, cmap="gray")
    plt.plot(border_pixels[:, 1], border_pixels[:, 0], "-b")
    plt.plot(border_pixels[:, 1], border_pixels[:, 0], ".r")

    spe_A = cross_ratio_spectrum(template_im)
    spe_B = cross_ratio_spectrum(template_im_test)
    spe_C = cross_ratio_spectrum(C)
    spe_D = cross_ratio_spectrum(D)
    spe_E = cross_ratio_spectrum(E)
    spe_F = cross_ratio_spectrum(F)
    spe_G = cross_ratio_spectrum(G)
    spe_H = cross_ratio_spectrum(H)
    spe_J = cross_ratio_spectrum(J)
    spe_K = cross_ratio_spectrum(K)
    spe_L = cross_ratio_spectrum(L)
    spe_M = cross_ratio_spectrum(M)
    spe_N = cross_ratio_spectrum(N)
    spe_O = cross_ratio_spectrum(O)
    spe_P = cross_ratio_spectrum(P)
    spe_G_1 = cross_ratio_spectrum(G_1)
    spe_G_2 = cross_ratio_spectrum(G_2)
    spe_G_3 = cross_ratio_spectrum(G_3)
    list_1 = [spe_G_1,spe_G_2,spe_G_3]
    
    result_list = []
    for i in range(len(list_1)):
        a = 0
        b = 0
        c = 0
        for j in range(200):
            if list_1[i][j] == -1:
                a += 1
            elif list_1[i][j] == 0:
                b += 1
            else:
                c += 1
        result_list.append([a,b,c])

    answer = []
    for i in range(len(result_list)):
        if result_list[i][1] == 0:
            if result_list[i][0]*3 < result_list[i][2]:
                answer.append(D)
            elif result_list[i][0]*3 >= result_list[i][2]:
                answer.append("G")
            else:
                answer.append("O")
        elif result_list[i][1] <= 20:     
            if  result_list[i][1] < 10:
                answer.append("B")
            else:
                answer.append("P")  
        elif result_list[i][0] > result_list[i][2] and result_list[i][0] > result_list[i][1]:
            if result_list[i][1] == 0 :
                answer.append("I")
            else :
                answer.append("L")    
        elif result_list[i][2] > result_list[i][1] and result_list[i][2] > result_list[i][0]:    
            if result_list[i][2] < 100:
                if result_list[i][0] > result_list[i][1]:
                    answer.append("A")
                else:
                    answer.append("N")
            elif result_list[i][2] >150 and result_list[i][2] < 180:
                answer.append("E")
            elif result_list[i][2] >150 and result_list[i][2] < 180:   
                if result_list[i][0] > result_list[i][1]:
                    answer.append("C")
                elif result_list[i][0]*2 > result_list[i][1]: 
                    answer.append("H")
                else:
                    answer.append("K")   
            else :
                answer.append("M")
        elif  result_list[i][1] > result_list[i][2] and result_list[i][1] > result_list[i][0]:   
            if result_list[i][2] > result_list[i][1] :
                answer.append("F")
            else:
                answer.append("J") 
    print("character of license plate :" ,answer[0]+answer[1]+answer[2])                     
    list = [spe_A,spe_B,spe_C,spe_D,spe_E,spe_F,spe_G,spe_H,spe_J,spe_K,spe_L,spe_M,spe_N,spe_O,spe_P] 
    result_list = []


if __name__ == "__main__":
    main() 
                     