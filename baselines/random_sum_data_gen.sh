for rest_flag in "rest" "full"
do
  for length in "150" "300"
  do
#    file="/cs/scratch/chenghao/"
#    if test -f "$file"; then
#        echo "$file exists. skipping.."
#    else
#        echo "$file not exists. running.."
        echo "Generating data for $rest_flag - $length"
        python demo_load_jsonl_dataset_in_hf.py ${rest_flag} ${length}
#    fi
  done
done