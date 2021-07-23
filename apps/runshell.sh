q#!/bin/bash

file1='./graph_names.txt'
file2='./bellman-ford/compare/bellman_originaldistance.txt'
file3='./dijkstra/compare/dijkstra_originaldistance.txt'


echo "vertexNum,edgeNum,iterationNum,maxEdgeDegree,minProcessEdge,maxProcessEdge,percentage,sOriginalDistance,sMinEdgetoProcess,sMaxEdgetoProcess,sPartialGraphProcess,sReduceExecution,sAtomicBlock,Error,executionTime" >> '../analysis-results/bellman_performance_results.csv'

while read line1;
do

cd ./bellman-ford/GPU/bellman-ford-base
make -f Makefile || exit
for n in {0..9};
do
	./run $line1 0
done
./run $line1 1 
cd ../../../


iter=$(head -n 1 $file2)


cd ./bellman-ford/GPU/bellman-ford-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter)
do
	for n in {0..9};
	do
		./run $line1 0 $ite
	done
	./run $line1 1 $ite
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-min
make -f Makefile || exit
for min in {1..7};
do
	for n in {0..9};
	do
		./run $line1 0 $min
	done
	./run $line1 1 $min
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-max
make -f Makefile || exit
for t in {1..5};
do
	for n in {0..9};
	do
		./run $line1 0 $t
	done
	./run $line1 1 $t
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-partial
make -f Makefile || exit
for p in 0.25 0.5 0.75 0.9:
do
	for n in {0..9};
	do
		./run $line1 0 $p
	done
	./run $line1 1 $p
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-min-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for min in {1..7};
	do
		for n in {0..9};
		do
			./run $line1 0 $ite $min
		done
		./run $line1 1 $ite $min
	done
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-max-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for t in {1..5};
	do
		for n in {0..9};
		do
			./run $line1 0 $ite $t
		done
		./run $line1 1 $ite $t
	done
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-partial-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter)
do
	for p in 0.25 0.5 0.75 0.9:
	do
		for n in {0..9};
		do
			./run $line1 0 $ite $p
		done
		./run $line1 1 $ite $p
	done
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-min-max
make -f Makefile || exit
for min in {1..7};
do
	for max in {1..5}
	do
		for n in {0..9};
		do
			./run $line1 0 $min $max
		done
		./run $line1 1 $min $max
	done
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-min-partial
make -f Makefile || exit
for min in {1..7};
do
	for p in 0.25 0.5 0.75 0.9:
	do
		for n in {0..9};
		do
			./run $line1 0 $min $p
		done
		./run $line1 1 $min $p
	done
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-max-partial
make -f Makefile || exit
for t in {1..5};
do
	for p in 0.25 0.5 0.75 0.9:
	do
		for n in {0..9};
		do
			./run $line1 0 $t $p
		done
		./run $line1 1 $t $p
	done
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-min-max-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for min in {1..7};
	do
		for max in {1..5}
		do
			for n in {0..9};
			do
				./run $line1 0 $ite $min $max
			done
			./run $line1 1 $ite $min $max
		done
	done
done
cd ../../../



cd ./bellman-ford/GPU/bellman-ford-min-partial-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for min in {1..7};
	do
		for p in 0.25 0.5 0.75 0.9:
		do
			for n in {0..9};
			do
				./run $line1 0 $ite $min $p
			done
			./run $line1 1 $ite $min $p
		done
	done
done
cd ../../../



cd ./bellman-ford/GPU/bellman-ford-max-partial-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for t in {1..5};
	do
		for p in 0.25 0.5 0.75 0.9:
		do
			for n in {0..9};
			do
				./run $line1 0 $ite $t $p
			done
			./run $line1 1 $ite $t $p
		done
	done
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-min-max-partial
make -f Makefile || exit
for min in {1..7};
do
	for max in {1..5}
	do
		for p in 0.25 0.5 0.75 0.9:
		do
			for n in {0..9};
			do
				./run $line1 0 $min $max $p
			done
			./run $line1 1 $min $max $p
		done
	done
done
cd ../../../


cd ./bellman-ford/GPU/bellman-ford-mix-appr
make -f Makefile || exit
for ite in $(seq 0 $iter)
do
	for min in {1..7};
	do
		for max in {1..5}
		do
			for p in 0.25 0.5 0.75 0.9:
			do
				for n in {0..9};
				do
					./run $line1 0 $ite $min $max $p
				done
				./run $line1 1 $ite $min $max $p
			done
		done
	done
done
cd ../../../


done < $file1



echo "vertexNum,edgeNum,iterationNum,maxEdgeDegree,minProcessEdge,maxProcessEdge,percentage,sOriginalDistance,sMinEdgetoProcess,sMaxEdgetoProcess,sPartialGraphProcess,sReduceExecution,sAtomicBlock,Error,executionTime" >> '../analysis-results/dijkstra_performance_results.csv'

while read line1;
do

cd ./dijkstra/GPU/dijkstra-base
make -f Makefile || exit
for n in {0..9};
do
	./run $line1 0
done
./run $line1 1 
cd ../../../


iter=$(head -n 1 $file3)


cd ./dijkstra/GPU/dijkstra-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter)
do
	for n in {0..9};
	do
		./run $line1 0 $ite
	done
	./run $line1 1 $ite
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-min
make -f Makefile || exit
for min in {1..7};
do
	for n in {0..9};
	do
		./run $line1 0 $min
	done
	./run $line1 1 $min
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-max
make -f Makefile || exit
for t in {1..5};
do
	for n in {0..9};
	do
		./run $line1 0 $t
	done
	./run $line1 1 $t
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-partial
make -f Makefile || exit
for p in 0.25 0.5 0.75 0.9:
do
	for n in {0..9};
	do
		./run $line1 0 $p
	done
	./run $line1 1 $p
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-min-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for min in {1..7};
	do
		for n in {0..9};
		do
			./run $line1 0 $ite $min
		done
		./run $line1 1 $ite $min
	done
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-max-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for t in {1..5};
	do
		for n in {0..9};
		do
			./run $line1 0 $ite $t
		done
		./run $line1 1 $ite $t
	done
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-partial-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter)
do
	for p in 0.25 0.5 0.75 0.9:
	do
		for n in {0..9};
		do
			./run $line1 0 $ite $p
		done
		./run $line1 1 $ite $p
	done
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-min-max
make -f Makefile || exit
for min in {1..7};
do
	for max in {1..5}
	do
		for n in {0..9};
		do
			./run $line1 0 $min $max
		done
		./run $line1 1 $min $max
	done
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-min-partial
make -f Makefile || exit
for min in {1..7};
do
	for p in 0.25 0.5 0.75 0.9:
	do
		for n in {0..9};
		do
			./run $line1 0 $min $p
		done
		./run $line1 1 $min $p
	done
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-max-partial
make -f Makefile || exit
for t in {1..5};
do
	for p in 0.25 0.5 0.75 0.9:
	do
		for n in {0..9};
		do
			./run $line1 0 $t $p
		done
		./run $line1 1 $t $p
	done
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-min-max-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for min in {1..7};
	do
		for max in {1..5}
		do
			for n in {0..9};
			do
				./run $line1 0 $ite $min $max
			done
			./run $line1 1 $ite $min $max
		done
	done
done
cd ../../../



cd ./dijkstra/GPU/dijkstra-min-partial-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for min in {1..7};
	do
		for p in 0.25 0.5 0.75 0.9:
		do
			for n in {0..9};
			do
				./run $line1 0 $ite $min $p
			done
			./run $line1 1 $ite $min $p
		done
	done
done
cd ../../../



cd ./dijkstra/GPU/dijkstra-max-partial-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for t in {1..5};
	do
		for p in 0.25 0.5 0.75 0.9:
		do
			for n in {0..9};
			do
				./run $line1 0 $ite $t $p
			done
			./run $line1 1 $ite $t $p
		done
	done
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-min-max-partial
make -f Makefile || exit
for min in {1..7};
do
	for max in {1..5}
	do
		for p in 0.25 0.5 0.75 0.9:
		do
			for n in {0..9};
			do
				./run $line1 0 $min $max $p
			done
			./run $line1 1 $min $max $p
		done
	done
done
cd ../../../


cd ./dijkstra/GPU/dijkstra-mix-appr
make -f Makefile || exit
for ite in $(seq 0 $iter)
do
	for min in {1..7};
	do
		for max in {1..5}
		do
			for p in 0.25 0.5 0.75 0.9:
			do
				for n in {0..9};
				do
					./run $line1 0 $ite $min $max $p
				done
				./run $line1 1 $ite $min $max $p
			done
		done
	done
done
cd ../../../


done < $file1


echo "vertexNum,edgeNum,iterationNum,maxEdgeDegree,minProcessEdge,maxProcessEdge,percentage,sOriginalDistance,sMinEdgetoProcess,sMaxEdgetoProcess,sPartialGraphProcess,sReduceExecution,sAtomicBlock,Error,executionTime" >> '../analysis-results/hybrid_performance_results.csv'

while read line1;
do

cd ./hybrid/GPU/hybrid-base
make -f Makefile || exit
for n in {0..9};
do
	./run $line1 0
done
./run $line1 1 
cd ../../../


iter=$(head -n 1 $file3)


cd ./hybrid/GPU/hybrid-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter)
do
	for n in {0..9};
	do
		./run $line1 0 $ite
	done
	./run $line1 1 $ite
done
cd ../../../


cd ./hybrid/GPU/hybrid-min
make -f Makefile || exit
for min in {1..7};
do
	for n in {0..9};
	do
		./run $line1 0 $min
	done
	./run $line1 1 $min
done
cd ../../../


cd ./hybrid/GPU/hybrid-max
make -f Makefile || exit
for t in {1..5};
do
	for n in {0..9};
	do
		./run $line1 0 $t
	done
	./run $line1 1 $t
done
cd ../../../


cd ./hybrid/GPU/hybrid-partial
make -f Makefile || exit
for p in 0.25 0.5 0.75 0.9:
do
	for n in {0..9};
	do
		./run $line1 0 $p
	done
	./run $line1 1 $p
done
cd ../../../


cd ./hybrid/GPU/hybrid-min-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for min in {1..7};
	do
		for n in {0..9};
		do
			./run $line1 0 $ite $min
		done
		./run $line1 1 $ite $min
	done
done
cd ../../../


cd ./hybrid/GPU/hybrid-max-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for t in {1..5};
	do
		for n in {0..9};
		do
			./run $line1 0 $ite $t
		done
		./run $line1 1 $ite $t
	done
done
cd ../../../


cd ./hybrid/GPU/hybrid-partial-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter)
do
	for p in 0.25 0.5 0.75 0.9:
	do
		for n in {0..9};
		do
			./run $line1 0 $ite $p
		done
		./run $line1 1 $ite $p
	done
done
cd ../../../


cd ./hybrid/GPU/hybrid-min-max
make -f Makefile || exit
for min in {1..7};
do
	for max in {1..5}
	do
		for n in {0..9};
		do
			./run $line1 0 $min $max
		done
		./run $line1 1 $min $max
	done
done
cd ../../../


cd ./hybrid/GPU/hybrid-min-partial
make -f Makefile || exit
for min in {1..7};
do
	for p in 0.25 0.5 0.75 0.9:
	do
		for n in {0..9};
		do
			./run $line1 0 $min $p
		done
		./run $line1 1 $min $p
	done
done
cd ../../../


cd ./hybrid/GPU/hybrid-max-partial
make -f Makefile || exit
for t in {1..5};
do
	for p in 0.25 0.5 0.75 0.9:
	do
		for n in {0..9};
		do
			./run $line1 0 $t $p
		done
		./run $line1 1 $t $p
	done
done
cd ../../../


cd ./hybrid/GPU/hybrid-min-max-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for min in {1..7};
	do
		for max in {1..5}
		do
			for n in {0..9};
			do
				./run $line1 0 $ite $min $max
			done
			./run $line1 1 $ite $min $max
		done
	done
done
cd ../../../



cd ./hybrid/GPU/hybrid-min-partial-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for min in {1..7};
	do
		for p in 0.25 0.5 0.75 0.9:
		do
			for n in {0..9};
			do
				./run $line1 0 $ite $min $p
			done
			./run $line1 1 $ite $min $p
		done
	done
done
cd ../../../



cd ./hybrid/GPU/hybrid-max-partial-reduce
make -f Makefile || exit
for ite in $(seq 1 $iter);
do
	for t in {1..5};
	do
		for p in 0.25 0.5 0.75 0.9:
		do
			for n in {0..9};
			do
				./run $line1 0 $ite $t $p
			done
			./run $line1 1 $ite $t $p
		done
	done
done
cd ../../../


cd ./hybrid/GPU/hybrid-min-max-partial
make -f Makefile || exit
for min in {1..7};
do
	for max in {1..5}
	do
		for p in 0.25 0.5 0.75 0.9:
		do
			for n in {0..9};
			do
				./run $line1 0 $min $max $p
			done
			./run $line1 1 $min $max $p
		done
	done
done
cd ../../../


cd ./hybrid/GPU/hybrid-mix-appr
make -f Makefile || exit
for ite in $(seq 0 $iter)
do
	for min in {1..7};
	do
		for max in {1..5}
		do
			for p in 0.25 0.5 0.75 0.9:
			do
				for n in {0..9};
				do
					./run $line1 0 $ite $min $max $p
				done
				./run $line1 1 $ite $min $max $p
			done
		done
	done
done
cd ../../../


done < $file1
