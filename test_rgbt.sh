for ((runid=15; runid>=10; runid--))
do
   python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mosstrack --yaml_name mosstrack_256_full --dataset_name LasHeR --threads 16 --num_gpus 4 --epoch $runid  && echo "Command for runid $runid executed successfully"
done

for ((runid=15; runid>=10; runid--))
do
    python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mosstrack --yaml_name mosstrack_256_full --dataset_name RGBT234 --threads 16 --num_gpus 4 --epoch $runid  && echo "Command for runid $runid executed successfully"
done

do
    python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mosstrack --yaml_name mosstrack_256_full_VTUAV --dataset_name VTUAVST --threads 16 --num_gpus 4 --epoch $runid  && echo "Command for runid $runid executed successfully"
done
