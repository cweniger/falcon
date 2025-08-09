#for net_type in gf nice sospf maf naf; do
for net_type in nice; do
    rm -rf /scratch-local/cweniger/sim_dir /scratch-local/cweniger/graph_dir
    falcon launch wandb.project=falcon_tmp2 wandb.group=zuko_${net_type}_v24 graph.z.module_config.net_type=zuko_${net_type} \
	    directories.sim_dir=/scratch-local/cweniger/sim_dir \
	    directories.graph_dir=/scratch-local/cweniger/graph_dir
done
