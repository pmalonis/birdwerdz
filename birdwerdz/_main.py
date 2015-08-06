#!/usr/bin/env python
import argparse
from birdwerdz import hdf
import inspect
import re

def func2parser(func, description_sep="Parameters", arg_sep=":", subparsers=None):
    '''Converts a function object to an argparse argument parser. The doc string
    is processed to supply the arguments to the argument parser
    
    Parameters
    ----------
    func : function to convert to argument parser
    description_sep : string that separates general help from argument specific help

    arg_sep : string that separates argument name from argument help

    subparsers : subparsers object. If given, a subparser corresponding to the 
    given function will be added to the subparsers object, and the subparsers object  
    will be returned. Otherwise, a single argument parser will be returned
 
    Returns
    -------
    parser : Argument parser or subparsers object that will be added to 
    
    arg_dict : dictionary whose keys are argument names and values are 
    argparse argument objects
    '''

    description = func.__doc__.split(description_sep)[0]
    if subparsers:
        parser = subparsers.add_parser(func.__name__,
                                       help=description,
                                       formatter_class=
                                       argparse.ArgumentDefaultsHelpFormatter)
    else:
        parser = argparse.ArgumentParser(prog=func.__name__,
                                         description=description,
                                         formatter_class=
                                         argparse.ArgumentDefaultsHelpFormatter)
    arg_help = func.__doc__.split(arg_sep)[1:]
    
    for i in xrange(len(arg_help)-1): #removes argument names from help string
        arg_help[i] = ' '.join(arg_help[i].split()[:-1])
 
    args,_,_,defaults = inspect.getargspec(func)
    if not defaults: defaults = ()
    flag_list=[]
    arg_dict=dict()
    for idx, (arg, h) in enumerate(zip(args,arg_help)):
        default_idx = idx-(len(args)-len(defaults))
        if default_idx < 0:
            arg_name=(arg,)
            keywords = {'help':h}
        else:
            arg_name=('--' + arg,)
            if arg[0] not in flag_list:
                arg_name = ('-' + arg[0],) + arg_name
                flag_list.append(arg[0])
            keywords = {'default': defaults[default_idx],
                        'type': type(defaults[default_idx]),
                        'help': h}
        arg_dict[arg] = parser.add_argument(*arg_name, **keywords)

    return parser, arg_dict
    
def main():
    parser = argparse.ArgumentParser(prog="birdwerdz",
                                     description = """Automated
                                     birdsong recognition""")
    subparsers = parser.add_subparsers()
    for _, member in inspect.getmembers(hdf):
        if inspect.isfunction(member) and member.__module__ == 'birdwerdz.hdf':
            p,arg_dict=func2parser(member, subparsers=subparsers)
            p.set_defaults(func=member)
            for arg_name, arg in arg_dict.items():
                if arg_name in ('clusters','labels'):
                    arg.nargs = '+'
                    arg.type = int

    options = parser.parse_args()
    args = {key:value for key,value in options.__dict__.iteritems() if key != 'func'} 
    options.func(**args)
    
if __name__=='__main__':
    main()
