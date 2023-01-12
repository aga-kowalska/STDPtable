"""
This is the code for lab1.  It's python.  
If code is inside these triple quotes, it's a comment.
You can also have a comment from a #, but it's just
until the end of the line.
"""

#import the cod for the pyNN nest interface.
import pyNN.nest as sim

#set up some constants in all caps for my style.
RUN_DURATION = 100
TIME_STEP = 0.1

numberNeurons = 1

#Anything that starts with a def is a function.
#Remember python uses indenting for blocks.  So this one ends at the
#next def.
def makeDCSource():
    clamp =  sim.DCSource(amplitude=0.8, start=20.0, stop=80.0)
    return (clamp)

#Python functions don't need to return anything.
def makeSynapses(pre_synaptic_neuron, post_synaptic_neuron):
    #Make a spike source
    stdp_model = sim.STDPMechanism(
            timing_dependence = sim.SpikePairRule(tau_plus=5.0, tau_minus=5.0, A_plus = 0.0001, A_minus = 0.0001),
            weight_dependence = sim.MultiplicativeWeightDependence(w_min = 0.0, w_max=0.01), 
            weight = 0.0,
            delay = TIME_STEP,
            dendritic_delay_fraction = float(1))
   
    #make a synapse
    firingWt = 0.2
    connector = []
    connector = connector + [(0,0,firingWt,TIME_STEP)]
    fromListConnector = sim.FromListConnector(connector)
    synapse = sim.Projection(pre_synaptic_neuron, post_synaptic_neuron,  fromListConnector, synapse_type=stdp_model)
    return synapse
    

#----main---
sim.setup(timestep=TIME_STEP,min_delay=TIME_STEP, max_delay=TIME_STEP, debug=0)

post_synaptic_neuron=sim.Population(numberNeurons,sim.IF_cond_exp,  cellparams={})
pre_synaptic_neuron=sim.Population(1,sim.IF_cond_exp,cellparams={})


post_synaptic_neuron.record({'spikes'})

dCSource = makeDCSource()
pre_synaptic_neuron.inject(dCSource)

synapse = makeSynapses(pre_synaptic_neuron, post_synaptic_neuron)

sim.run(RUN_DURATION)

post_synaptic_neuron.write_data("post_synaptic_stdp_Spikes.pkl",'spikes')

synapseWeight = synapse.get(["weight"], format="list")

print(synapseWeight)
