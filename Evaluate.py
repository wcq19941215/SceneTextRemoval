import numpy as np
from scipy import signal, ndimage
from math import floor
import gauss


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    size = min(img1.shape[0], 11)
    sigma = 1.5
    window = gauss.fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = signal.fftconvolve(img1, window, mode = 'valid')
    mu2 = signal.fftconvolve(img2, window, mode = 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode = 'valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode = 'valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode = 'valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))


def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image 
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity 
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on 
    Signals, Systems and Computers, Nov. 2003 
    
    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2)) / 4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(im1, im2, cs_map = True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                mode = 'reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                mode = 'reflect')
        im1 = filtered_im1[: : 2, : : 2]
        im2 = filtered_im2[: : 2, : : 2]

    # Note: Remove the negative and add it later to avoid NaN in exponential.
    sign_mcs = np.sign(mcs[0 : level - 1])
    sign_mssim = np.sign(mssim[level - 1])
    mcs_power = np.power(np.abs(mcs[0 : level - 1]), weight[0 : level - 1])
    mssim_power = np.power(np.abs(mssim[level - 1]), weight[level - 1])
    return np.prod(sign_mcs * mcs_power) * sign_mssim * mssim_power
    #return (np.prod(mcs[0 : level - 1] ** weight[0 : level - 1]) * (mssim[level - 1] ** weight[level - 1]))

def mae(img1, img2):
    r = np.asarray(img1, dtype=np.float64).ravel()
    print(r.shape)
    c = np.asarray(img2, dtype=np.float64).ravel()
    return np.mean(np.abs(r - c))/255


def PeakSignaltoNoiseRatio(origImg, distImg, max_value=255):
    origImg = origImg.astype(float)
    distImg = distImg.astype(float)

    M, N = np.shape(origImg)
    error = origImg - distImg
    MSE = sum(sum(error * error)) / (M * N)

    if MSE > 0:
        PSNR = 10 * np.log10(max_value * max_value / MSE)
    else:
        PSNR = 99
    # print(PSNR)
    # print(MSE)

    return PSNR , MSE


def cqm(orig_img, dist_img):
    M, N, C = np.shape(orig_img)
    
    if C != 3:
        CQM = float("inf")
        return CQM
        
    Ro = orig_img[:, :, 0]
    Go = orig_img[:, :, 1]
    Bo = orig_img[:, :, 2]

    Rd = dist_img[:, :, 0]
    Gd = dist_img[:, :, 1]
    Bd = dist_img[:, :, 2]

    ################################################
    ###       Reversible YUV Transformation      ###
    ################################################
    YUV_img1 = np.zeros((M, N, 3))
    YUV_img2 = np.zeros((M, N, 3))

    for i in range(M):
        for j in range(N):
            ### Original Image Trasnformation  ###
            # Y=(R+2*G+B)/4
            YUV_img1[i, j, 0] = floor((Ro[i, j] + Go[i, j] * 2 + Bo[i, j]) / 4)
            YUV_img2[i, j, 0] = floor((Rd[i, j] + Gd[i, j] * 2 + Bd[i, j]) / 4)
            # U=R-G
            YUV_img1[i, j, 1] = max(0, Ro[i, j] - Go[i, j])
            YUV_img2[i, j, 1] = max(0, Rd[i, j] - Gd[i, j])
            # V=B-G
            YUV_img1[i, j, 2] = max(0, Bo[i, j] - Go[i, j])
            YUV_img2[i, j, 2] = max(0, Bd[i, j] - Gd[i, j])          

    ################################################
    ###               CQM Calculation            ###
    ################################################
    Y_psnr = PeakSignaltoNoiseRatio(YUV_img1[:, :, 0], YUV_img2[:, :, 0]); # PSNR for Y channel
    U_psnr = PeakSignaltoNoiseRatio(YUV_img1[:, :, 1], YUV_img2[:, :, 1]); # PSNR for U channel
    V_psnr = PeakSignaltoNoiseRatio(YUV_img1[:, :, 2], YUV_img2[:, :, 2]); # PSNR for V channel

    CQM = (Y_psnr * 0.9449) + (U_psnr + V_psnr) / 2 * 0.0551

    return CQM


def Evaluate(GT, BC):
    print(np.shape(GT))
    [M, N, C] = np.shape(GT)
    dimension = M * N

    GT = np.ndarray((M, N, 3), 'u1', GT.tostring()).astype(float)
    BC = np.ndarray((M, N, 3), 'u1', BC.tostring()).astype(float)

    if C == 3:      # In case of color images, use luminance in YCbCr
        R = GT[:, :, 0]
        G = GT[:, :, 1]
        B = GT[:, :, 2]

        YGT = .299 * R + .587 * G + .114 * B

        R = BC[:, :, 0]
        G = BC[:, :, 1]
        B = BC[:, :, 2]

        YBC = .299 * R + .587 * G + .114 * B

    else:
        YGT = GT
        YBC = BC

    ############################# AGE ########################################
    Diff = abs(YGT - YBC).round().astype(np.uint8)
    AGE = np.mean(Diff)

    ########################### EPs and pEPs #################################
    threshold = 20

    Errors = Diff > threshold
    EPs = sum(sum(Errors)).astype(float)
    pEPs = EPs / float(dimension)

    ########################## CEPs and pCEPs ################################
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    erodedErrors = ndimage.binary_erosion(
        Errors, structure).astype(Errors.dtype)
    CEPs = sum(sum(erodedErrors))
    pCEPs = CEPs / float(dimension)

    ############################# MSSSIM #####################################
    MSSSIM = msssim(YGT, YBC)
    # print("MSSSIM",MSSSIM)
    SSIM = np.mean(ssim(YGT, YBC))
    # print("SSIM",SSIM)
    MAE = mae(GT, BC)
    ############################# PSNR #######################################
    PSNR,MSE = PeakSignaltoNoiseRatio(YGT, YBC)
    
    ############################# CQM ########################################
    # if C == 3:
    #     CQM = cqm(GT, BC)

    return (AGE, pEPs, pCEPs, MSSSIM, PSNR, SSIM, MSE,MAE)
