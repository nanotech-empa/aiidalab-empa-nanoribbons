cd ~
wget https://polybox.ethz.ch/index.php/s/C4YaQ1zsmWXctk2/download
mv download SSSP_modified.tar.gz
tar xvf SSSP_modified.tar.gz
rm SSSP_modified.tar.gz

verdi data upf uploadfamily SSSP_modified SSSP_modified "SSSP modified"

