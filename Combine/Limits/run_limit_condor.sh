#!/bin/bash

year="$1"
suffix="$2"

statonly="$3"

path_dir="/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/CombineResults/bkg"



# if [ "$year" = "2024" ]; then
# 	path_dir="${path_dir}/SignalExtraction/${year}"
# else
# 	path_dir="${path_dir}/${year}"
# fi

path_dir="${path_dir}/${year}"

input_dir="${path_dir}/workspace${suffix}"
output_dir="${path_dir}/limits${suffix}"

if [ -d "$output_dir" ]; then
	echo "Directory already exists: $output_dir"
else
	mkdir -p "$output_dir"
fi

if [[ "$statonly" -eq 1 ]]; then
	output_dir="${path_dir}/limits_v5_nosys"
fi

templates=(1) # Only unrolled here

submission_file="$input_dir/condor_submit.sh"
submission_local="$input_dir/run_local.sh"

echo "#!/bin/bash" > "$submission_file"

for temp in "${templates[@]}"; do  

	if [ "$year" = "2022" ] || [ "$year" = "2022EE" ] || [ "$year" = "2023" ] || [ "$year" = "2023BPiX" ]; then
		files="$input_dir/workspace_XYH_4b_${temp}_13p6TeV_${year}_XtoYHto4B_MX-*_MY-*.root"
    else
    	files="$input_dir/workspace_XYH_4b_${temp}_13p6TeV_${year}_XtoYHto4B_Par-MX-*-MY-*.root"
    
    fi
    #files="$input_dir/workspace_XYH_4b_${temp}_13p6TeV_${year}_XtoYHto4B_Par-MX-*-MY-*.root"

    for file in $files; do
        if [[ ! -f "$file" ]]; then
            echo "No files matching pattern for template $temp"
            continue
        fi

        file_name=$(basename "$file")
	echo $file_name

        comb=$(echo "$file_name" | sed -n 's/.*4b_\([0-9]*\)_.*/\1/p')
        #mx_value=$(echo "$file_name" | sed -n 's/.*MX-\([0-9]*\)_MY-[0-9]*.*/\1/p')
        #my_value=$(echo "$file_name" | sed -n 's/.*MX-[0-9]*_MY-\([0-9]*\).*/\1/p')
	if [ "$year" = "2022" ] || [ "$year" = "2022EE" ] || [ "$year" = "2023" ] || [ "$year" = "2023BPiX" ]; then
		mx_value=$(echo "$file_name" | sed -n 's/.*MX-\([0-9]*\)_MY-[0-9]*.*/\1/p')
		my_value=$(echo "$file_name" | sed -n 's/.*MX-[0-9]*_MY-\([0-9]*\).*/\1/p')	
	else
		mx_value=$(echo "$file_name" | sed -n 's/.*MX-\([0-9]*\)-MY-[0-9]*.*/\1/p')
		my_value=$(echo "$file_name" | sed -n 's/.*MX-[0-9]*-MY-\([0-9]*\).*/\1/p')
	fi
        if [[ -n "$mx_value" && -n "$my_value" ]]; then

            output_file="$input_dir/execute_${comb}_${mx_value}_${my_value}.csh"
            #output_name="${comb}_${mx_value}_${my_value}"
	    output_name="_Comb_${comb}_MX_${mx_value}_MY_${my_value}"

	    combine_command="combine -t -1 $file -n $output_name # --rMin -100 --rMax 100"
	    if [[ "$statonly" -eq 1 ]]; then
		    combine_command="$combine_command --freezeParameters allConstrainedNuisances"
	    fi

            cat << EOF > "$output_file"
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd \$CMSSW_BASE/src
eval \`scramv1 runtime -sh\`

cd $input_dir
$combine_command
mv $input_dir/higgsCombine$output_name.AsymptoticLimits.mH120.root  $output_dir
EOF

            chmod +x "$output_file"
            echo "Created script: $output_file"

	    output_file_sub="$input_dir/submit_${comb}_${mx_value}_${my_value}.sh"

	    cat << EOF > "$output_file_sub"
universe = vanilla
executable = $output_file
getenv = TRUE
log = $input_dir/job$output_name.log
output = $input_dir/job$output_name.out
error = $input_dir/job$output_name.err
notification = never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
queue
+MaxRuntime = 100000
EOF
	echo "Created submission script: $output_file_sub"


	echo "condor_submit $output_file_sub" >>  $submission_file
	echo "sh $output_file" >> $submission_local

	   fi
   done

done

chmod +x "$submission_file"
chmod +x "$submission_local"
echo "All jobs for template $temp prepared. Submit with: bash $submission_file"
