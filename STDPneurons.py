#import the code for the pyNN nest interface.
import pyNN.nest as sim

RUN_DURATION = 100
TIME_STEP = 0.01
INITIAL_WEIGHT = 0.01

def makeDCSource(inputWeight):
    base = 0.7
    increment = 0.1
    myAmplitude = base + (increment * inputWeight)
    clamp =  sim.DCSource(amplitude=myAmplitude, start=20.0, stop=80.0)
    return clamp

def makeSynapses(pre_synaptic_neuron, post_synaptic_neuron):
    stdp_model = sim.STDPMechanism(
            timing_dependence = sim.SpikePairRule(tau_plus=5.0, tau_minus=5.0, A_plus = 0.0001, A_minus = 0.0001),
            weight_dependence = sim.MultiplicativeWeightDependence(w_min = 0.0, w_max=0.01), 
            weight = 0.0,
            delay = TIME_STEP,
            dendritic_delay_fraction = float(1))
   
    #make a synapse
    firingWt = INITIAL_WEIGHT 
    connector = []
    connector = connector + [(0,0,firingWt,TIME_STEP)]
    fromListConnector = sim.FromListConnector(connector)
    synapse = sim.Projection(pre_synaptic_neuron, post_synaptic_neuron,  fromListConnector, synapse_type=stdp_model)
    return synapse
    
def makeSpikeSource(spikeTime, post_synaptic_neuron):
    inputSpikeTime = [spikeTime]
    spikeArray = {'spike_times':[inputSpikeTime]}
    spikeGen = sim.Population(1, sim.SpikeSourceArray, spikeArray)
    firingWt = 0.1
    connector = [(0, 0, firingWt, TIME_STEP)]
    fromListConnector = sim.FromListConnector(connector)
    sim.Projection(spikeGen,  post_synaptic_neuron, fromListConnector, receptor_type='excitatory')


#----main---
sim.setup(timestep=TIME_STEP,min_delay=TIME_STEP, max_delay=TIME_STEP, debug=0)

post_synaptic_neuron=sim.Population(1,sim.IF_cond_exp,  cellparams={})
pre_synaptic_neuron=sim.Population(1,sim.IF_cond_exp,cellparams={})


post_synaptic_neuron.record({'spikes'})
pre_synaptic_neuron.record({'spikes'})
dCSource = makeDCSource(2)
pre_synaptic_neuron.inject(dCSource)

synapse = makeSynapses(pre_synaptic_neuron, post_synaptic_neuron)
makeSpikeSource(05.0, post_synaptic_neuron)
sim.run(RUN_DURATION)

post_synaptic_neuron.write_data("post_synaptic_stdp_Spikes.pkl",'spikes')
pre_synaptic_neuron.write_data("pre_synaptic_stdp_Spikes.pkl",'spikes')
synapseWeight = synapse.get(["weight"], format="list")

print("Synaptic weight. Initial: ", INITIAL_WEIGHT, " Final: ", synapseWeight)
