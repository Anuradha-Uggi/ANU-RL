
####### P30k ###### T2
ckpts=("resnet50_epoch(39)_step(25760)_R1[0.8718]_R5[0.9362].ckpt" "resnet50_epoch(31)_step(20352)_R1[0.8894]_R5[0.9450].ckpt" "resnet50_epoch(15)_step(10016)_R1[0.8892]_R5[0.9475].ckpt" "resnet50_epoch(23)_step(13824)_R1[0.8957]_R5[0.9485].ckpt" "resnet50_epoch(35)_step(18612)_R1[0.8945]_R5[0.9485].ckpt" "resnet50_epoch(15)_step(7024)_R1[0.8898]_R5[0.9437].ckpt" "resnet50_epoch(19)_step(7200)_R1[0.8938]_R5[0.9457].ckpt")


sizes=(2 3 4 5 6 7 8)
indices=(0 1 2 3 4 5 6)
#sizes=(0.9 1.0)
for k in "${indices[@]}"
do 
   ckpt="${ckpts[$k]}"
   i="${sizes[$k]}"
   echo "Index: $i, Value: $ckpt"    

done


