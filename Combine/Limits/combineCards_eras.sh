#!/bin/bash
combine_years=($1) # (2024 2025)

suffix=$2

# Eras to combine
# combine_years=(2024 2025)

# Base directory
base_dir="/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/CombineResults/bkg"

# Datacard directory inside each year
card_dir="datacards${suffix}"

# first year as reference
ref_year="${combine_years[0]}"

input_dir_1="${base_dir}/${ref_year}/${card_dir}"

comb_year_dir="combined"
for i in "${combine_years[@]}"; do
	comb_year_dir+="_${i}"
done

# Output directory for combined datacards
output_subdir="${comb_year_dir}"

output_dir="${base_dir}/${output_subdir}/${card_dir}"
mkdir -p "$output_dir"

mkdir -p "${base_dir}/${output_subdir}/workspace${suffix}"
mkdir -p "${base_dir}/${output_subdir}/limits${suffix}"

templates=(1)



# Loop over templates
for temp in "${templates[@]}"; do
	
	# Loop over datacards for each template
	for file in "$input_dir_1"/XYH_4b_${temp}_*.txt; do

		if [[ ! -e "$file" ]]; then
            echo "ERROR: No datacards found matching XYH_4b_${temp}_*.txt in $input_dir_1"
            continue
        fi

    		file_name=$(basename "$file")

    		combine_cmd="combineCards.py"

    		# Add datacards from all years
    		for year in "${combine_years[@]}"; do

        		# card_path="${base_dir}/${year}/${card_dir}/${file_name}"
				target_file_name="${file_name//$ref_year/$year}"
            	card_path="${base_dir}/${year}/${card_dir}/${target_file_name}"

        		# Check if datacard exists
        		if [[ -f "$card_path" ]]; then
            			combine_cmd+=" era${year}=${card_path}"
        		else
            			echo "WARNING: Missing file $card_path"
        		fi
    		done
	
		# Output combined datacard
		out_file_name="${file_name//$ref_year/${comb_year_dir}}"
        output_card="${output_dir}/${out_file_name}"
		# output_card="${output_dir}/${file_name}"
		echo "Creating combined card: $output_card"

		#combine cards
		# echo ${combine_cmd}
		eval "${combine_cmd} > \"${output_card}\""

	done
done

# use: bash combineCards_eras.sh "2024 2025" _v4_Sys_WPBTag