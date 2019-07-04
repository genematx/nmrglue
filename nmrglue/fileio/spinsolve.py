"""
Functions for reading and writing Spinsolve binary (1d/par) files.
"""

from __future__ import print_function, division

__developer_info__ = """
Spinsolve file format information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To be added....

"""

from functools import reduce
import os
from warnings import warn

import numpy as np

from . import fileiobase
from ..process import proc_base


# data creation

def create_data(data):
    """
    Create a bruker data array (recast into a complex128 or int32)
    """
    if np.iscomplexobj(data):
        return np.array(data, dtype='complex128')
    else:
        return np.array(data, dtype='int32')


# universal dictionary functions

def guess_udic(dic, data, strip_fake=False):
    """
    Guess parameters of universal dictionary from dic, data pair.

    Parameters
    ----------
    dic : dict
        Dictionary of Spinsolve parameters.
    data : ndarray
        Array of NMR data.
    strip_fake: bool
        If data is proceed (i.e. read using `bruker.read_pdata`) and the Spinsolve
        processing parameters STSI and/or STSR are set, the returned sweep
        width and carrier frequencies is changed to values that are incorrect
        but instead can are intended to trick the normal unit_conversion object
        into producing the correct result.

    Returns
    -------
    udic : dict
        Universal dictionary of spectral parameters.

    """
    return udic


def add_axis_to_udic(udic, dic, udim, strip_fake):
    """
    Add axis parameters to a udic.

    Parameters
    ----------
    udic : dict
        Universal dictionary to update, modified in place.
    dic : dict
        Spinsolve dictionary used to determine axes parameters.
    dim : int
        Universal dictionary dimension to update.
    strip_fake: bool
        See `bruker.guess_udic`

    """
    return udic


def create_dic(c0, f0, dt, tau, nt, NS=1):
    """
    Create a Spinsolve parameter dictionary from a universal dictionary.

    Parameters
    ----------
    udic : dict
        Universal dictionary of spectral parameters.

    Returns
    -------
    dic : dict
        Dictionary of Spinsolve parameters.

    """

    dic = {}

    dic['Solvent'] = ''
    dic['Sample'] = ''
    dic['startTime'] = ''  # datestr(datetime('now','TimeZone','local','Format','d-MMM-y_HH:mm:ss Z')),
    dic['acqDelay'] = tau*1e+06     # Ringdown delay in ms
    dic['b1Freq'] = c0              # B1 frequency in MHz
    dic['bandwidth'] = (1/dt)/1000  # Sweep bandwidth in kHz
    dic['dwellTime'] = 1000*dt      # Dwell time in ms
    dic['experiment'] = "1D"
    dic['expName'] = "1D"
    dic['nrPnts'] = nt
    dic['nrScans'] = NS
    dic['repTime'] = 0              # Repetiotion time in ms
    dic['rxChannel'] = "1H"
    dic['rxGain'] = 0               # Reciever gain in dB
    dic['lowestFrequency'] = (-(1/dt)/2+f0)       # Lowest frequency in Hz
    dic['totalAcquisitionTime'] = 53          # Total acquisition time in sec
    dic['graphTitle'] = '1D-1H-\"StandardScan\"'
    dic['userData'] = ''
    dic['90Amplitude'] = 0          # Amplitude of the 90-degree pulse in dB
    dic['pulseLength'] = 0          # Pulse length in ms
    dic['Protocol'] = "1D PROTON"
    dic['Options'] = 'Scan(StandardScan)'
    dic['Spectrometer'] = 'Python'
    dic['Software'] = 'Python'

    return dic

# Global read/write function and related utilities

def read(dir, bin_file=None, pars_files=None):
    """
    Read Spinsolve files from a directory.

    Parameters
    ----------
    dir : str
        A dictionary (or any file in the directory) to read from.
    bin_file : str, optional
        Filename of binary (.1d) file in directory. None uses standard files.
    pars_file : str, optional
        List of filename(s) of .par parameter files in directory. None uses
        standard files.

    Returns
    -------
    dic : dict
        Dictionary of Spinsolve parameters.
    data : ndarray
        Array of NMR data.

    See Also
    --------
    read_pdata : Read Spinsolve processed files.
    read_lowmem : Low memory reading of Spinsolve files.
    write : Write Spinsolve files.

    """

    # If the dir is actually a file
    if os.path.isfile(dir):
        dir = os.path.dirname(dir)

    # Read parameter file(s) and update the dictionary
    dic = {}
    if pars_files is None: pars_files = ["acqu.par", "protocol.par"]
    for f in pars_files:
        if os.path.isfile(os.path.join(dir, f)):
            dic.update(read_pars(os.path.join(dir, f)))

    # read the binary file
    if bin_file is None: bin_file = 'data.1d'
    if os.path.isfile(os.path.join(dir, bin_file)):
        data_dic, data = read_binary(os.path.join(dir, bin_file))
        dic.update(data_dic)


    # read the pulse program and add to the dictionary

    # determine shape and complexity for direct dim if needed

    return dic, data

def write(dir, dic, data, big=True, overwrite=False):
    """
    Write Spinsolve files to disk.

    Parameters
    ----------
    dir : str
        Directory to write files to.
    dir : dict
        Dictionary of Spinsolve parameters.
    data : array_like
        Array of NMR data
    overwrite : bool, optional
        Set True to overwrite files, False will raise a Warning if files exist.
    big : bool
        Endiness to write binary data with True of big-endian, False for
        little-endian.

    See Also
    --------
    read : Read Spinsolve files.

    """

    # write out the .par file
    write_pars( os.path.join(dir, 'acqu.par'), dic, overwrite=overwrite)

    # write out the binary data
    write_binary(os.path.join(dir, 'data.1d'), dic, data, big=big, overwrite=overwrite)
    return


# Spinsolve binary (fid/ser) reading and writing

def read_binary(filename, big=False):
    """
    Read Spinsolve binary data from file and return dic,data pair.


    Parameters
    ----------
    filename : str
        Filename of Spinsolve binary file.
    big : bool
        Endianness of binary file, True for big-endian, False for
        little-endian.

    Returns
    -------
    dic : dict
        Dictionary containing "FILE_SIZE" key and value.
    data : ndarray
        Array of raw NMR data.

    See Also
    --------
    read_binary_lowmem : Read binary file using minimal memory.

    """
    # open the file and get the data
    with open(filename, 'rb') as f:
        head = (f.read(3*4).decode('ascii'), f.read(1*4))                                   # Read 3+1 4-byte blocks
        size = np.frombuffer(f.read(4*4), dtype='int32')     # Read another 4 4-byte blocks; each of them is an integer
        data = np.frombuffer(f.read(), dtype='float32')

    # create dictionary
    dic = {"FILE_SIZE": os.stat(filename).st_size}

    # Reshape the data according to the read sizes of array dimensions
    if size[1] == 1:
        t = data[:size.prod()]
        data = (data[size.prod()::2] - 1j*data[size.prod()+1::2]).reshape(size[:2])
    else:
        data =  (data[::2] - 1j*data[1::2]).reshape(size[:2], order='F')

    return dic, data


def write_binary(filename, dic, data, overwrite=False, big=True):
    """
    Write Spinsolve binary data to file.

    Parameters
    ----------
    filename : str
        Filename to write to.
    dic : dict
        Dictionary of Spinsolve parameters.
    data : ndarray
        Array of NMR data.
    overwrite : bool
        True to overwrite files, False will raise a Warning if file exists.
    big : bool
        Endiness to write binary data with True of big-endian, False for
        little-endian.

    See Also
    --------
    write_binary_lowmem : Write Spinsolve binary data using minimal memory.

    """
    # open the file for writing
    f = fileiobase.open_towrite(filename, overwrite=overwrite)

    # Write the header
    n1, n2, n3, n4 = data.size, 1, 1, 1     # Dimensions of the data array
    dt = dic['dwellTime'] / 1000
    t = np.linspace(0, (n1-1)*dt, n1, dtype='float32')
    data = np.concatenate([t, data.ravel().real, -data.ravel().imag]).astype('float32')
    head = np.array([1347571539, 1145132097, 1446063665, 504, n1, n2, n3, n4], dtype='int32')

    put_data(f, data, big)
    f.close()
    return

# Reading / writing Spinsolve parameter files

def read_pars(filename):
    """
    Read a Spinsolve parameters .par file into a dictionary.


    Parameters
    ----------
    filename : str
        Filename of a Spinsolve .par file.

    Returns
    -------
    dic : dict
        Dictionary of parameters in file.

    See Also
    --------
    write_pars : Write a Spinsolve .par file.

    """
    dic = {}  # create empty dictionary

    with open(filename, 'r') as f:
        while True:     # loop until end of file is found

            line = f.readline().rstrip()    # read a line
            if line == '':      # end of file found
                break

            else:
                line = line.split('=')
                key = line[0].strip()
                val = line[1].strip()
                if val[0] == "\"":
                    val = val.strip("\"")
                else:
                    try:
                        val = float(val)
                    except ValueError: pass

                dic[key] = val

    return dic

def write_pars(filename, dic, overwrite=False):
    """
    Writes a Spinsolve parameters .par file.


    Parameters
    ----------
    filename : str
        Filename of a Spinsolve .par file.

    dic : dict
        Dictionary of parameters.

    See Also
    --------
    read_pars : Read a Spinsolve .par file.

    """

    # open the file for writing
    f = fileiobase.open_towrite(filename, overwrite=overwrite, mode='w')

    # write out the core headers
    newline=''
    for k, v in dic.items():
        f.write(newline+"{:<25} = {}".format(k, v))
        newline = '\n'

    # close the file
    f.close()

def put_data(f, data, big=True):
    """
    Put data to file object with given endiness.
    """
    if big:
        f.write(data.astype('>i4').tostring())
    else:
        f.write(data.astype('<i4').tostring())
    return

def _merge_dict(a, b):
    c = a.copy()
    c.update(b)
    return c
