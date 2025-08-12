#for net_type in gf nice sospf maf naf; do
for net_type in nice; do
    rm -rf /scratch-local/cweniger/sim_dir /scratch-local/cweniger/graph_dir
    falcon swarm logging.project=falcon_tmp4 logging.group=zuko_${net_type}_v1 graph.z.estimator.net_type=zuko_${net_type} \
	    paths.buffer=/scratch-local/cweniger/sim_dir \
	    paths.graph=/scratch-local/cweniger/graph_dir
#    falcon sample prior wandb.project=falcon_tmp3 wandb.group=zuko_${net_type}_v2 graph.z.estimator.net_type=zuko_${net_type} \
#	    directories.sim_dir=/scratch-local/cweniger/sim_dir \
#	    directories.graph_dir=/scratch-local/cweniger/graph_dir
#    falcon sample posterior logging.project=falcon_tmp4 logging.group=zuko_${net_type}_v1 graph.z.estimator.net_type=zuko_${net_type} \
#	    paths.buffer=/scratch-local/cweniger/sim_dir \
#	    paths.graph=/scratch-local/cweniger/graph_dir
done
