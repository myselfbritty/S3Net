import os
import glob

keyword = "S3NET_folds_withsegm_removed"
root="./"
config_folder = os.path.join(root+'configs/S3NET_S1_2/')
work_dir_label = os.path.join(root+"pre-trained-weights/Stage_1_2")
result_root_folder ="./S3NET_outputs/"
keyword1_for_weight = "mask_rcnn_r50_fpn_1x_EndoVis17"
epoch = 12
result_folder = os.path.join(result_root_folder, keyword)
if not os.path.exists(result_root_folder):
        os.mkdir(result_root_folder)
if not os.path.exists(result_folder):
        os.mkdir(result_folder)
raw_dumps = os.path.join(result_folder, 'raw_dumps')
if not os.path.exists(raw_dumps):
        os.mkdir(raw_dumps)

for folder in sorted(os.listdir(config_folder)):
        fn = folder[-1]
        for epochs in sorted(os.listdir(os.path.join(config_folder,folder))):
                if not os.path.exists(os.path.join(raw_dumps, folder)):
                        os.mkdir(os.path.join(raw_dumps, folder))
                raw_dumps_fold = os.path.join(raw_dumps, folder)
                path_to_config = os.path.join(config_folder,folder+"/"+epochs)
                path_to_workdirs = os.path.join(work_dir_label, folder,'epoch_'+str(epoch)+'.pth')
                path_to_output_dirs = os.path.join(raw_dumps_fold, keyword1_for_weight+"_" +folder+"_"+keyword+"_ep_"+str(epoch)+".pkl")
                path_to_text_write = os.path.join(raw_dumps_fold, keyword1_for_weight+"_" +folder+"_"+keyword+"_ep_"+str(epoch)+".txt")
                print("Command:","CUDA_VISIBLE_DEVICES=0 python tools/test.py "+path_to_config + " "+ path_to_workdirs+ " --out "+ path_to_output_dirs+ " --eval " + " segm " + " >> " + path_to_text_write)
                os.system("CUDA_VISIBLE_DEVICES=0 python tools/test.py "+path_to_config+ " " +path_to_workdirs+" --out "+ path_to_output_dirs+ " --eval " + " segm " + " >> " + path_to_text_write)
                
