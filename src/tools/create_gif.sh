 
#!/bin/bash
# for i in {1..20}
# do
#  echo "create gif of exp" $i
#  python gif_maker.py --path2folder="/mnt/hdd/rkato_outputs/blenshape_interpolation/PCA_trained3dgs/${i}_comp/400015"
#  gifsicle -i /mnt/hdd/rkato_outputs/blenshape_interpolation/PCA_trained3dgs/${i}_comp/400015/exp.gif -O3 --colors 128 -o /mnt/hdd/rkato_outputs/blenshape_interpolation/PCA_trained3dgs/${i}_comp/400015/exp-opt.gif
#  python gif_maker.py --path2folder="/mnt/hdd/rkato_outputs/blenshape_interpolation/PCA_trained3dgs/${i}_comp/400048"
#  gifsicle -i /mnt/hdd/rkato_outputs/blenshape_interpolation/PCA_trained3dgs/${i}_comp/400048/exp.gif -O3 --colors 128 -o /mnt/hdd/rkato_outputs/blenshape_interpolation/PCA_trained3dgs/${i}_comp/400048/exp-opt.gif
# done

# for i in {1..20}
# do
#  echo "create gif of exp" $i
#  python gif_maker.py --path2folder="/mnt/hdd/rkato_outputs/blenshape_interpolation/SLDC_trained3dgs_0.20.3100/${i}_comp/400015"
#  gifsicle -i /mnt/hdd/rkato_outputs/blenshape_interpolation/SLDC_trained3dgs_0.20.3100/${i}_comp/400015/exp.gif -O3 --colors 128 -o /mnt/hdd/rkato_outputs/blenshape_interpolation/SLDC_trained3dgs_0.20.3100/${i}_comp/400015/exp-opt.gif
#  python gif_maker.py --path2folder="/mnt/hdd/rkato_outputs/blenshape_interpolation/SLDC_trained3dgs_0.20.3100/${i}_comp/400048"
#  gifsicle -i /mnt/hdd/rkato_outputs/blenshape_interpolation/SLDC_trained3dgs_0.20.3100/${i}_comp/400048/exp.gif -O3 --colors 128 -o /mnt/hdd/rkato_outputs/blenshape_interpolation/SLDC_trained3dgs_0.20.3100/${i}_comp/400048/exp-opt.gif
# done

for i in {1..20}
do
 echo "create gif of exp" $i
 python blendshape_demo.py --path2folder="/mnt/hdd/rkato_outputs/blenshape_interpolation/SLDC_trained3dgs_0.20.3100/${i}_comp/400015"
 python blendshape_demo.py --path2folder="/mnt/hdd/rkato_outputs/blenshape_interpolation/SLDC_trained3dgs_0.20.3100/${i}_comp/400048"
 python blendshape_demo.py --path2folder="/mnt/hdd/rkato_outputs/blenshape_interpolation/PCA_trained3dgs/${i}_comp/400015"
 python blendshape_demo.py --path2folder="/mnt/hdd/rkato_outputs/blenshape_interpolation/PCA_trained3dgs/${i}_comp/400048"
done

