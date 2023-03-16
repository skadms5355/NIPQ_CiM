
mapping_mode='2T2R'
pbits='8 7 6 5 4 3 2'
co_noise='0.05'
tnipq='qhwnoise'
tnoise_type='prop'
tco_noise='0.03'

echo ${mapping_mode}
python script_main_nipq_noise.py --psum_comp -g $1 --mapping_mode ${mapping_mode} --pbits $pbits --co_noise ${co_noise} --tnipq $tnipq --tnoise_type $tnoise_type --tco_noise $tco_noise