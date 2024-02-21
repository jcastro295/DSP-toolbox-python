#!bin/bash

file_names=("python_data.zip")

# download each file via wget
for i in ${!file_names[@]};
do
    echo "Downloading ${file_names[i]}..."
    wget "https://www.eecis.udel.edu/~jcastro/data/${file_names[0]}" -O "${file_names[0]}"

    echo "Unzipping ${file_names[i]}..."
    unzip ${file_names[i]}

    echo "Removing ${file_names[i]}..."
    rm ${file_names[i]}
done
echo "Done!"