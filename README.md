# GPUProject

##### Graph generation code:

```
octave --persist kron_generator.m
kron_generator(scale,edge_factor,"path/to/example_graphs/graph_name.txt")
```

<ul>
	<li>
		scale: can start from 18 and go to 26 (if memory allows it)
	</li>
	<li>
		edge_factor: can be stable at 16
	</li>
	<li>
		graph_name: should be as follow: graph_26_16_pos-10.txt
	</li>
</ul>


##### Then, in ./apps/ file, there is a script to run the program

```
bash runshell.sh
```
