import os
import Evaluate
import scipy
# import tkMessageBox
from tkinter import messagebox
import sys
# import ipdb
def Utility(GT_path, evaluated_path,num_path):
    '''
    Function to evaulate the your resutls for SBMnet dataset, this code will generate a 'cm.txt' file in your result path to save all the metrics.
    input:  GT_path: the path of the groundtruth folder.
    evaluated_path: the path of your evaluated results folder.
    '''
    result_file = os.path.join(evaluated_path, 'cm.txt')

    with open(result_file, 'w') as fid:
        fid.write('\t\timage_name\tPSNR\tSSIM\tMSE\tMAE\r\n')

    m_AGE = 0
    m_pEPs = 0
    m_pCEPs = 0
    m_MSSSIM = 0
    m_PSNR = 0
    m_SSIM = 0
    m_MSE = 0
    m_MAE = 0
    # ipdb.set_trace()
    
    c_AGE = 0
    c_pEPs = 0
    c_pCEPs = 0
    c_MSSSIM = 0
    c_PSNR = 0
    c_SSIM = 0
    c_MSE = 0
    c_MAE = 0
        
    image_num = 0
    
    for root, dirs, files in os.walk(evaluated_path):
        MSSSIM_max = 0
        for i in files:
            # 判断是否以.jpg结尾
            if i.endswith('.JPG') or i.endswith('.jpg') or i.endswith('.PNG') or i.endswith('.png'):
                picname = i.split('.')[0]
                print("picname:",picname)
                num = picname.split('_')[0]
            
                #if more than one GT exists for the video, we keep the
                #metrics with the highest MSSSIM value.
                if(num_path==2000):
                    GT_img = scipy.misc.imread(GT_path+num+".jpg")       #background ground truth
                    result_img = scipy.misc.imread(evaluated_path+num+".jpg")
                if(num_path==1080):
                    GT_img = scipy.misc.imread(GT_path+num+"_gt.png")       #background ground truth
                    result_img = scipy.misc.imread(evaluated_path+picname+".png")
               
                AGE, pEPs, pCEPs, MSSSIM, PSNR, SSIM, MSE,MAE = Evaluate.Evaluate(GT_img, result_img);
                if MSSSIM > MSSSIM_max:
                    MSSSIM_max = MSSSIM
                v_AGE = AGE
                v_pEPs = pEPs
                v_pCEPs = pCEPs
                v_MSSSIM = MSSSIM
                v_PSNR = PSNR
                v_SSIM = SSIM
                v_MSE = MSE
                v_MAE = MAE
                
                
                #save the video evaluation results
                with open(result_file, 'a+') as fid:
                    fid.write('\t\t' + picname + ':\t' + str(round(v_PSNR, 4)) + '\t' + str(round(v_SSIM, 4)) + '\t' + str(round(v_MSE, 4)) + '\t' + str(round(v_MAE, 4)) + '\r\n')

                c_AGE = c_AGE + v_AGE
                c_pEPs = c_pEPs + v_pEPs
                c_pCEPs = c_pCEPs + v_pCEPs
                c_MSSSIM = c_MSSSIM + v_MSSSIM
                c_PSNR = c_PSNR + v_PSNR
                c_SSIM = c_SSIM + v_SSIM
                c_MSE = c_MSE + v_MSE
                c_MAE = c_MAE + v_MAE
                image_num = image_num + 1

        c_AGE = c_AGE / float(image_num)
        c_pEPs = c_pEPs / float(image_num)
        c_pCEPs = c_pCEPs / float(image_num)
        c_MSSSIM = c_MSSSIM / float(image_num)
        c_PSNR = c_PSNR / float(image_num)
        c_SSIM = c_SSIM / float(image_num)
        c_MSE = c_MSE / float(image_num)
        c_MAE = c_MAE / float(image_num)

        #save the category evaluation results
        with open(result_file, 'a+') as fid:
            fid.write('\t\timage_name\tPSNR\tSSIM\tMSE\tMAE\r\n')
            fid.write('\r\n' + 'gt' + '_AVG::\t\t' + str(round(c_PSNR, 4)) + '\t' + str(round(c_SSIM, 4)) + '\t' + str(round(c_MSE, 4))+ '\t' + str(round(c_MAE, 4)) + '\r\n\r\n')

        m_AGE = m_AGE + c_AGE
        m_pEPs = m_pEPs + c_pEPs
        m_pCEPs = m_pCEPs + c_pCEPs
        m_MSSSIM = m_MSSSIM + c_MSSSIM
        m_PSNR = m_PSNR + c_PSNR
        m_SSIM = m_SSIM + c_SSIM
        m_MSE = m_MSE + c_MSE
        m_MAE = m_MAE + c_MAE

   

    with open(result_file, 'a+') as fid:
        fid.write('Total:\t\t\t' + str(round(m_PSNR, 4)) + '\t' + str(round(m_SSIM*100, 4)) + '\t' + str(round(m_MSE, 4)) + '\t' + str(round(m_MAE*100, 4)) + '\r\n')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python {0} <GT_path> <evaluated_path>".format(sys.argv[0]))
    num_path=2000
    GT_path = '/data/cqwang/64_backup/cqwang/dataset/IJCIA_dataset/test/test_{}_256/gt/'.format(str(num_path))
    evaluated_path = './result_{}/'.format(str(num_path))
    Utility(GT_path, evaluated_path,num_path)

    num_path=1080
    GT_path = '/data/cqwang/64_backup/cqwang/dataset/IJCIA_dataset/test/test_{}_256/gt/'.format(str(num_path))
    evaluated_path = './result_{}/'.format(str(num_path))
    Utility(GT_path, evaluated_path,num_path)
