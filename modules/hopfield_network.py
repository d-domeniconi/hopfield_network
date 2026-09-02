import matplotlib.image as mimg
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import itertools
from scipy.stats import norm


def train(num_neurons,patterns):
  
  """
    Train a Hopfield network using Hebbian learning.

    Constructs the synaptic weight matrix by summing the outer products
    of the stored patterns, normalized by the number of neurons.
    Self-connections are explicitly removed.

    Parameters
    ----------
    num_neurons : int
        Number of neurons in the network.
    patterns : list of ndarray
        List of bipolar patterns (values ±1) to be stored,
        each of shape (num_neurons,).

    Returns
    -------
    weights : ndarray
        Symmetric weight matrix of shape (num_neurons, num_neurons)
        with zero diagonal.
    """

  weights = np.zeros((num_neurons, num_neurons))
  for pattern in patterns:
    pattern_outer_product = np.outer(pattern, pattern)

    weights += pattern_outer_product / num_neurons

  np.fill_diagonal(weights, 0)
  return weights

def activate(value):
  
    """
    Binary activation function for Hopfield neurons.

    Implements a deterministic sign activation rule.
    Zero is mapped to +1.

    Parameters
    ----------
    value : float
        Local field acting on a neuron.

    Returns
    -------
    int
        +1 if value >= 0, otherwise -1.
    """

    return 1 if value >= 0 else -1

def retrieve(num_neurons,weights,initial_pattern,t_MAX=10000):
  
  """
  Retrieve a stored pattern using synchronous Hopfield dynamics.

  All neurons are updated simultaneously at each iteration using
  the state from the previous time step. The dynamics stop when
  a fixed point is reached or when the maximum number of iterations
  is exceeded.

  Parameters
  ----------
  num_neurons : int
      Number of neurons in the network.
  weights : ndarray
      Weight matrix of shape (num_neurons, num_neurons).
  initial_pattern : ndarray
      Initial state of the network (possibly noisy).
  t_MAX : int, optional
      Maximum number of iterations (default is 10000).

  Returns
  -------
  ndarray
      Final retrieved pattern after convergence.
  """

  X = np.copy(initial_pattern)
  stable = False

  t=0
  X_prev = X.copy()
  while (not stable) and (t < t_MAX):
    t += 1
    stable = True

    for i in range(num_neurons):
      X[i] = activate(np.dot(weights[i], X_prev))

      if X_prev[i] != X[i]:
        stable = False

    X_prev = X.copy()

  return X

def retrieve_async(num_neurons,weights,initial_pattern,t_MAX=100000):
  
  """
  Retrieve a stored pattern using asynchronous Hopfield dynamics.

  At each step, a single randomly chosen neuron is updated.
  The process stops after num_neurons consecutive updates
  without state changes, indicating convergence.

  Parameters
  ----------
  num_neurons : int
      Number of neurons in the network.
  weights : ndarray
      Weight matrix of shape (num_neurons, num_neurons).
  initial_pattern : ndarray
      Initial state of the network (possibly noisy).
  t_MAX : int, optional
      Maximum number of update steps (default is 10000).

  Returns
  -------
  ndarray
      Final retrieved pattern after convergence.
  """

  X = np.copy(initial_pattern)

  t = 0
  T = 0

  while (t < t_MAX):
    X_prev = X.copy()
    t += 1
    i = np.random.randint(num_neurons)
    X[i] = activate(np.dot(weights[i], X))

    if np.array_equal(X, X_prev):
       T += 1

       if T == num_neurons:
          break

    else:
       T = 0

  return X

def visualize_pattern(pattern, title):
  
  """
  Visualize a 1D Hopfield pattern as a bar plot.

  Parameters
  ----------
  pattern : ndarray
      Bipolar pattern to visualize.
  title : str
      Title of the plot.
  """

  plt.figure(figsize=(8, 2))
  plt.bar(range(len(pattern)), pattern)
  plt.title(title)
  plt.xlabel("Neuron Index")
  plt.ylabel("Activation")
  plt.ylim([-1.1, 1.1])
  plt.grid(axis='y')

def patterns(num_padroes, tamanho_padrao):
    """
    Gera padrões binários aleatórios compostos por -1 e 1.
    
    Parâmetros:
    num_padroes (int): Quantidade de padrões na lista (ex: P memórias).
    tamanho_padrao (int): Comprimento de cada padrão (ex: N neurônios/spins).
    
    Retorna:
    numpy.ndarray: Matriz onde cada linha é um padrão armazenado.
    """
    # Escolhe aleatoriamente entre -1 e +1 para preencher a matriz
    padroes = np.random.choice([-1, 1], size=(num_padroes, tamanho_padrao))
    
    # Caso precise estritamente de uma lista nativa do Python (lista de listas):
    # return padroes.tolist()
    
    return padroes

def add_noise(pattern, noise_level):
    """Add noise to a pattern by flipping a fraction of its bits.

    Parameters
    ----------
    pattern : ndarray
        Original bipolar pattern (values -1 and 1).
    noise_level : float
        Fraction of neurons to flip (0 <= noise_level <= 1).

    Returns
    -------
    ndarray
        Noisy version of the input pattern.
    """
    # 1. Faz uma cópia para não alterar o padrão original na memória
    noisy = pattern.copy()

    # 2. Calcula quantos neurônios/bits serão invertidos
    n_flip = int(noise_level * len(pattern))

    # 3. Escolhe os índices aleatórios sem repetição
    flip_indices = np.random.choice(len(pattern), n_flip, replace=False)

    # 4. Inverte os valores selecionados (ex: 1 vira -1, e -1 vira 1)
    noisy[flip_indices] *= -1

    return noisy

def noisy_patterns(patterns_to_store,P, noise_level):

    """
    Generate noisy versions of stored patterns.

    Parameters
    ----------
    patterns_to_store : list of ndarray
        Original stored patterns.
    P : int
        Number of patterns.
    noise_level : float
        Fraction of bits flipped in each pattern.

    Returns
    -------
    list of ndarray
        List of noisy patterns.
    """

    noisy_pattern = []

    for i in range(P):
        noisy_pattern.append(add_noise(patterns_to_store[i], noise_level))

    return noisy_pattern

def retrieve_all_sync(noisy_patterns, num_neurons, weights, num_memories):
   
   """
   Retrieve multiple patterns using synchronous dynamics.

   Parameters
   ----------
   noisy_patterns : list of ndarray
       Initial noisy patterns.
   num_neurons : int
       Number of neurons.
   weights : ndarray
       Hopfield weight matrix.
   num_memories : int
       Number of patterns to retrieve.

   Returns
   -------
   list of ndarray
       Retrieved patterns.
   """

   retrieved_patterns = []

   for i in range(num_memories):
      initial_pattern = np.copy(noisy_patterns[i])
      retrieved_pattern = retrieve(num_neurons,weights,initial_pattern)
      retrieved_patterns.append(retrieved_pattern)

   return retrieved_patterns

def vizualize(original_patterns, noisy_patterns,retrieved_patterns, N, save, figlabel = "pattern", recall = True):

    """
    Visualize original, noisy, and retrieved patterns as images.

    Assumes that N is a perfect square so that patterns can be
    reshaped into square images.

    Parameters
    ----------
    original_patterns : list of ndarray
        Stored patterns.
    noisy_patterns : list of ndarray
        Noisy input patterns.
    retrieved_patterns : list ofweights = hp.train(len(vetor_ufsc), memories) ndarray
        Patterns after retrieval.
    N : int
        Number of neurons per pattern.

    Returns
    -------
    int
        Always returns 0.
    """

    if recall == True:    
        for i,(O,No,R) in enumerate(zip(original_patterns, noisy_patterns,retrieved_patterns)):
            matrix_O  =  O.reshape( int(math.sqrt(N)), int(math.sqrt(N)))
            matrix_No = No.reshape( int(math.sqrt(N)), int(math.sqrt(N)))
            matrix_R  =  R.reshape( int(math.sqrt(N)), int(math.sqrt(N)))
            fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3))
            ax[0].imshow(matrix_O , cmap='gray', interpolation='nearest')
            ax[1].imshow(matrix_No, cmap='gray', interpolation='nearest')
            ax[2].imshow(matrix_R , cmap='gray', interpolation='nearest')
            ax[0].set_title(f"Original", fontsize=20)
            ax[1].set_title(f"Noisy", fontsize=20)
            ax[2].set_title(f"Recalled", fontsize=20)
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            if save == True:
                fig.savefig(f"figures/{figlabel}_{i+1}.png",dpi = 600)
        plt.show()

    else:
        matrix_O  =  original_patterns[0].reshape( int(math.sqrt(N)), int(math.sqrt(N)))
        plt.imshow(matrix_O , cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    return None

def retrieve_all_assync(noisy_patterns, num_neurons, weights, num_memories):
   
   """
   Retrieve multiple patterns using asynchronous dynamics.

   Parameters
   ----------
   noisy_patterns : list of ndarray
       Initial noisy patterns.
   num_neurons : int
       Number of neurons.
   weights : ndarray
       Hopfield weight matrix.
   num_memories : int
       Number of patterns to retrieve.

   Returns
   -------
   list of ndarray
       Retrieved patterns.
   """

   retrieved_patterns = []

   for i in range(num_memories):
      initial_pattern = np.copy(noisy_patterns[i])
      retrieved_pattern = retrieve_async(num_neurons,weights,initial_pattern,t_MAX=1000000)
      retrieved_patterns.append(retrieved_pattern)

   return retrieved_patterns


def overlap(state, pattern):

    """
    Compute the overlap between a network state and a pattern.

    The overlap measures similarity and serves as an order parameter.

    Parameters
    ----------
    state : ndarray
        Current network state.
    pattern : ndarray
        Reference pattern.

    Returns
    -------
    float
        Normalized overlap value in [-1, 1].
    """

    N = len(state)
    return np.dot(state, pattern) / N

def current_activation_tendency(weights, current_state):

    """
    Compute the local field acting on each neuron.

    Parameters
    ----------
    weights : ndarray
        Weight matrix.
    current_state : ndarray
        Current network state.

    Returns
    -------
    ndarray
        Local field vector.
    """

    return np.dot(weights, current_state)

def current_energy(weights, current_state):

    """
    Compute the Hopfield energy of a given state.

    Parameters
    ----------
    weights : ndarray
        Weight matrix.
    current_state : ndarray
        Network state.

    Returns
    -------
    float
        Energy of the state.
    """

    K = current_activation_tendency(weights, current_state)
    return -0.5 * np.dot(current_state, K)

def all_binary_states(num_neurons):

    """
    Generate all possible binary states of a Hopfield network.

    Parameters
    ----------
    num_neurons : int
        Number of neurons.

    Returns
    -------
    list of ndarray
        List containing all 2^num_neurons bipolar states.
    """

    combos = itertools.product([-1, 1], repeat=num_neurons)
    return [np.array(state) for state in combos]

def energy_landscape(weights):

    """
    Compute the full energy landscape of a Hopfield network.

    Parameters
    ----------
    weights : ndarray
        Weight matrix.

    Returns
    -------
    tuple
        energies_sorted : ndarray
            Energies sorted in ascending order.
        states_sorted : list of ndarray
            Corresponding states sorted by energy.
    """

    num_neurons = weights.shape[0]
    states = all_binary_states(num_neurons)
    energies = [current_energy(weights, s) for s in states]
    energies_sorted = np.sort(energies)
    states_sorted = []
    for i,energy in enumerate(energies_sorted):
        for state in states:
            if (i+2)/2 == int((i+2)/2):
                if current_energy(weights, state) == energy:
                    states_sorted.append(state)

    return energies_sorted, states_sorted

def plot_energy_landscape(weights):

    """
    Plot the energy landscape of a Hopfield network.

    Parameters
    ----------
    weights : ndarray
        Weight matrix.
    """

    _, states_sorted = energy_landscape(weights)
    plt.figure(figsize=(8, 5))
    plt.plot([i for i in range(len(states_sorted))] ,[current_energy(weights, s) for s in states_sorted],'-o',linewidth = 1)
    plt.title(f"Hopfield Energy Landscape ({weights.shape[0]} neurons)")
    plt.xlabel("State index (sorted by energy)")
    plt.ylabel("Energy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def random_weights(N):

    """
    Generate a random symmetric weight matrix with zero diagonal.

    Parameters
    ----------
    N : int
        Number of neurons.

    Returns
    -------
    ndarray
        Symmetric random weight matrix.
    """

    W = np.random.randn(N, N)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)
    return W

def find_memory_and_anti_memory_indices(all_states, memories):

    """
    Identify indices of stored memories and their anti-memories.

    Parameters
    ----------
    all_states : list of ndarray
        All possible network states.
    memories : list of ndarray
        Stored memory patterns.

    Returns
    -------
    tuple
        memory_indices : list of int
        anti_memory_indices : list of int
    """

    memory_indices = []
    anti_memory_indices = []

    for i, state in enumerate(all_states):
        is_memory = False
        is_anti_memory = False
        for mem in memories:
            if np.array_equal(state, mem):
                is_memory = True
                break
            elif np.array_equal(state, -mem):
                is_anti_memory = True
                break

        if is_memory:
            memory_indices.append(i)
        elif is_anti_memory:
            anti_memory_indices.append(i)

    return memory_indices, anti_memory_indices

def sort_states(all_states, energies, memories):

    """
    Sort network states by energy, grouped around memories.

    Parameters
    ----------
    all_states : list of ndarray
        Network states.
    energies : list or ndarray
        Energy of each state.
    memories : list of ndarray
        Stored memory patterns.

    Returns
    -------
    list of ndarray
        Sorted states.
    """

    if memories is None or len(memories) == 0: 
        combined = sorted(zip(energies, all_states), key=lambda item: item[0])
        return [state for energy, state in combined]

    memory_indices, anti_memory_indices = find_memory_and_anti_memory_indices(all_states, memories)
    all_memory_indices = memory_indices + anti_memory_indices

    if not all_memory_indices: 
        combined = sorted(zip(energies, all_states), key=lambda item: item[0])
        return [state for energy, state in combined]

    nearest_memory_indices_map = {}
    for i, state in enumerate(all_states):
        distances = [abs(i - mem_idx) for mem_idx in all_memory_indices]
        nearest_memory_indices_map[i] = all_memory_indices[np.argmin(distances)]

    grouped_states = {mem_idx: [] for mem_idx in all_memory_indices}
    for i, state in enumerate(all_states):
        nearest_mem_idx = nearest_memory_indices_map[i]
        grouped_states[nearest_mem_idx].append((energies[i], state, i))

    sorted_states_list = []
    for nearest_mem_idx in sorted(all_memory_indices):
        states_in_group = grouped_states[nearest_mem_idx]

        states_before_nearest_memory = [(e, s, original_idx) for e, s, original_idx in states_in_group if original_idx < nearest_mem_idx]
        states_at_and_after_nearest_memory = [(e, s, original_idx) for e, s, original_idx in states_in_group if original_idx >= nearest_mem_idx]

        states_before_nearest_memory_sorted = [s for e, s, original_idx in sorted(states_before_nearest_memory, key=lambda item: item[0], reverse=True)]

        states_at_and_after_nearest_memory_sorted = [s for e, s, original_idx in sorted(states_at_and_after_nearest_memory, key=lambda item: item[0])]

        sorted_states_list.extend(states_before_nearest_memory_sorted)
        sorted_states_list.extend(states_at_and_after_nearest_memory_sorted)

    return sorted_states_list

def extract_memories(weights, all_states, energy_threshold):

    """
    Extract low-energy attractor states from the energy landscape.

    Parameters
    ----------
    weights : ndarray
        Weight matrix.
    all_states : list of ndarray
        All network states.
    energy_threshold : float
        Energy cutoff for selecting potential memories.

    Returns
    -------
    list of ndarray
        Unique memory candidates (up to sign).
    """

    potential_memories = []
    energies = [current_energy(weights, state) for state in all_states]

    for i, state in enumerate(all_states):
        
        if energies[i] < energy_threshold:
            potential_memories.append(state)

    
    unique_potential_memories = []
    for mem in potential_memories:
        is_unique = True
        for unique_mem in unique_potential_memories:
            if np.array_equal(mem, unique_mem) or np.array_equal(mem, -unique_mem):
                is_unique = False
                break
        if is_unique:
            unique_potential_memories.append(mem)


    return unique_potential_memories

def energy_table(M,axis):

    """
    Compute energies of fixed states over multiple random networks.

    Parameters
    ----------
    M : int
        Number of random weight realizations.
    axis : list of ndarray
        States for which energies are computed.

    Returns
    -------
    list of list of float
        Energies for each realization.
    """

    N = len(axis[0])

    L = []

    for _ in range(M):
        W = random_weights(N)

        L.append([current_energy(W, s) for s in list(axis)])

    return L

def mean_energies(M,axis):

    """
    Compute disorder-averaged energies of network states.

    Parameters
    ----------
    M : int
        Number of random networks.
    axis : list of ndarray
        Network states.

    Returns
    -------
    ndarray
        Mean energy of each state.
    """
    
    L = np.array(energy_table(M, axis))

    
    mean_energies = L.mean(axis=0)

    return mean_energies

def plot_energy_and_distribution(b, mean_energies_data, plot_color='blue'):

    """
    Plot average energy landscape and its distribution.

    Displays the mean energy per state alongside a logarithmic
    histogram of the energy distribution.

    Parameters
    ----------
    b : int or str
        Label identifying the number of stored patterns.
    mean_energies_data : ndarray
        Mean energies of network states.
    mu_val : float
        Mean of the fitted Gaussian (optional).
    sigma_val : float
        Standard deviation of the fitted Gaussian (optional).
    title_suffix : str
        Suffix used in plot labeling.
    plot_color : str, optional
        Color used in the plot (default is 'blue').
    """

    fig = plt.figure(figsize=(7.5, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[3, 1, 1, 1])

    # Main subplot for average energy plot
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.scatter(range(len(mean_energies_data)), mean_energies_data,s=20 , color=plot_color, label=f'{b}')
    ax1.legend(fontsize = 30)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    ax1.set_ylim(-8, 1)

    # Secondary subplot for rotated distribution plot (last column)
    ax2 = fig.add_subplot(gs[0, 3])
    counts, bins, _ = ax2.hist(mean_energies_data, bins=90, density=True, orientation='horizontal', color=plot_color, alpha=0.7, edgecolor='none', label='Histogram')

    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xscale('log')
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)
    
    plt.tight_layout()
    plt.savefig(f'Average_Visualization_2/(2)average_energy_{b}_trained_patterns.png',dpi = 600)
    plt.show()

def preprocessar_para_hopfield(img_path, size, threshold, swap_colors):
    """
    Loads an image and preprocesses it for use in Hopfield networks.

    The pipeline performs:

    1. Image loading from the given file path.
    2. Binary conversion using a threshold.
    3. Logical inversion to represent:
        - dark object as 1
        - light background as 0
    4. Detection of the active image region.
    5. Automatic cropping of useful content.
    6. Padding into a square canvas using the dominant border color.
    7. Resizing to a fixed resolution (size x size).
    8. Conversion to bipolar representation:
        -1 = background
        +1 = object
    9. Flattening into a vector for Hopfield input.
    10. Optional bipolar color inversion.

    Parameters
    ----------
    img_path : str
        Path to the image file.

    size : int
        Desired final square image size.
        Example: size=120 produces a 120x120 image.

    threshold : float
        Threshold used for binarization.
        Values above threshold become white.
        Values below threshold become black.

        Typical range:
        0.0 to 1.0

    swap_colors : bool
        If True, inverts the final bipolar encoding:
            +1 <-> -1

        Useful when the main object polarity is reversed.

    Returns
    -------
    img_final : numpy.ndarray
        2D matrix with shape (size, size) containing only {-1, +1} values.

    vector : numpy.ndarray
        Flattened 1D version of the final image.
        Shape = (size*size,)

    Notes
    -----
    This function is suitable for:

    - Hopfield networks
    - Associative memory
    - Binary pattern recognition
    - Logos, symbols, and characters

    Requirements
    ------------
    numpy
    matplotlib.image
    cv2
    """

    # -------------------------------------------------
    # 1. Converter para grayscale/binário
    # -------------------------------------------------
    img = mimg.imread(img_path)
    if img.ndim == 3:
        img_bw = (np.mean(img[:, :, :4], axis=2) > threshold).astype(float)
    else:
        img_bw = (img > threshold).astype(float)

    # Inverte: objeto preto vira 1, fundo branco vira 0
    img_bw = 1 - img_bw

    # -------------------------------------------------
    # 2. Encontrar conteúdo ativo
    # -------------------------------------------------
    rows, cols = np.where(img_bw == 1)

    top    = rows.min()
    bottom = rows.max()
    left   = cols.min()
    right  = cols.max()

    # Corte solicitado
    img_crop = img_bw[top:bottom+2, left:right+2]

    # -------------------------------------------------
    # 3. Descobrir cor dominante da borda do recorte
    # -------------------------------------------------
    borda = np.concatenate([
        img_crop[0, :],
        img_crop[-1, :],
        img_crop[:, 0],
        img_crop[:, -1]
    ])

    cor_borda = round(np.mean(borda))   # 0 ou 1

    # -------------------------------------------------
    # 4. Transformar em quadrado
    # -------------------------------------------------
    h, w = img_crop.shape
    lado = max(h, w)

    img_quad = np.ones((lado, lado)) * cor_borda

    y = (lado - h) // 2
    x = (lado - w) // 2

    img_quad[y:y+h, x:x+w] = img_crop

    # -------------------------------------------------
    # 5. Resize
    # -------------------------------------------------
    img_final = cv2.resize(
        img_quad,
        (size, size),
        interpolation=cv2.INTER_NEAREST
    )

    # -------------------------------------------------
    # 6. Converter para bipolar {-1,+1}
    # -------------------------------------------------
    img_final = 2 * img_final - 1

    # -------------------------------------------------
    # 7. Vetorizar
    # -------------------------------------------------
    vetor = img_final.flatten()

    if swap_colors == True:
        vetor = -vetor
        img_final = -img_final

    return img_final, vetor

def test_network(memory_list, weight_matrix, noise_level, synchronous=True, Save = True, label = "pattern"):
    """
    Tests a trained Hopfield network by adding noise to stored patterns,
    retrieving them, and visualizing the results.

    The procedure is:

    1. Apply random noise to the stored memory patterns.
    2. Recover the patterns using synchronous or asynchronous dynamics.
    3. Display original, noisy, and retrieved patterns side by side.

    Parameters
    ----------
    memory_list : list of numpy.ndarray
        List of stored bipolar memory patterns.

        Each pattern must be a 1D vector containing values such as:
        {-1, +1}

    weight_matrix : numpy.ndarray
        Trained Hopfield weight matrix with shape:

            (N, N)

        where N is the number of neurons.

    noise_level : float
        Fraction of bits/pixels to corrupt in each stored pattern.

        Typical range:
        0.0 to 1.0

        Examples:
        - 0.10 = 10% noise
        - 0.30 = 30% noise
        - 0.50 = 50% noise

    synchronous : bool, optional
        Determines which update rule is used during retrieval.

        If True:
            Uses synchronous updates, where all neurons are updated
            simultaneously.

        If False:
            Uses asynchronous updates, where neurons are updated
            one at a time.

        Default is True.

    Save : bool, optional
        If True, saves the visualizations of the patterns to disk.

    Returns
    -------
    None
        Displays the recovery results and returns nothing.

    Notes
    -----
    Internally, this function calls:

    - hp.noisy_patterns(...)
    - hp.retrieve_all_sync(...)
    - hp.retrieve_all_assync(...)
    - hp.vizualize(...)

    This function is useful for evaluating:

    - Memory robustness
    - Pattern recovery quality
    - Noise tolerance
    - Differences between synchronous and asynchronous dynamics

    Example
    -------
    test_network(memories, weights, noise_level=0.25)

    test_network(memories, weights, noise_level=0.25, synchronous=False)
    """
    noisies = noisy_patterns(memory_list, len(memory_list), noise_level=noise_level)
    if synchronous:
        retrieved_patterns = retrieve_all_sync(noisies, len(memory_list[0]), weight_matrix, len(memory_list))
    else:
        retrieved_patterns = retrieve_all_assync(noisies, len(memory_list[0]), weight_matrix, len(memory_list))
    vizualize(memory_list, noisies, retrieved_patterns, len(memory_list[0]), save=Save, figlabel=label)
    return None

def see_synapses(weights_matrix, save = True, figlabel = "Synaptic_Weights"):
    """
    Visualizes the synaptic weights of a Hopfield network as a heatmap.

    Parameters
    ----------
    weights_matrix : numpy.ndarray
        The weight matrix of the Hopfield network, typically of shape (N, N),
        where N is the number of neurons.

    save : bool, optional
        If True, saves the heatmap as a PNG file. Default is True.

    figlabel : str, optional
        The label for the figure file if saved. Default is "Synaptic_Weights".

    Returns
    -------
    None
        Displays the heatmap of synaptic weights and returns nothing.

    Notes
    -----
    This function uses matplotlib to create a heatmap representation of the
    synaptic weights. The color intensity indicates the strength of the weights,
    with darker colors representing stronger connections.

    Example
    -------
    see_synapses(weights)
    """
    #plt.style.use('dark_background')
    plt.imshow(weights_matrix, cmap='Grays_r')  # colormap can be changed
    cbar = plt.colorbar()
    cbar.set_label("Neuron Correlation", fontsize=18)  # Label for colorbar
    cbar.set_ticks([])  # Show colorbar ticks
    plt.axis('off')  # Hide axes
    plt.title("Brain's Synaptic Weights", fontsize=20)  # Title for the heatmap
    if save:
        plt.savefig(f"figures/{figlabel}.png", dpi=600)  # Save the figure with high resolution
    plt.show()
    return None

def plot_custom_energy_landscape(weights, state_axis, log = False):
    """
    Computes and plots the unsorted energy landscape along a custom state axis
    using high-resolution presentation styling.
    """
    energies = [current_energy(weights, s) for s in state_axis]
    
    plt.figure(figsize=(15, 6))
    plt.plot(energies, 'o-', linewidth=1.5, color='blue')
    plt.title(f'Energy Landscape for a Random Hopfield Network', fontsize=28)
    plt.xlabel(f'Network State {r"$\vec{\sigma}$"} (Initial Axis)', fontsize=28)
    plt.ylabel("Energy", fontsize=28)
    # plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    if log:
        plt.yscale('log')
    plt.tight_layout()
    plt.show()