# add completion for clustering
_clustering() 
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    mode="${COMP_WORDS[1]}"
    # opts for single file argument
    # nopts for no options
    # fopts for no autocompleion option (for numbers)
    
    # mode completion
    if [[ ${prev} == "clustering" ]]; then
        nopts="density network mpp coring noise filter stats"
    else
        # mode specific completion
        case "${mode}" in
            density)
                opts="--file --output --input --population --free-energy --free-energy-input --nearest-neighbors --nearest-neighbors-input" 
                nopts="--help --nthreads --verbose"
                fopts="--radius --radii --threshold-screening"
                ;;
            network)
                opts="--minpop --basename --output"
                nopts="--help --network-html --verbose"
                fopts="--min --max --step"
                ;;
            mpp)
                opts="--states --free-energy-input --concat-limits --tprob --output" 
                nopts="--help --verbose"
                fopts="--lagtime --qmin-from --qmin-to --qmin-step --concat-nframes --nthreads"
                ;;
            coring)
                opts="--states --windows --output --distrubtion --cores--concat-limits" 
                nopts="--help --verbose --iterative"
                fopts="--concat-nframes"
                ;;
            noise)
                opts="--states --output --basename --cores --concat-limits" 
                nopts="--help --verbose"
                fopts="--concat-nframes --cmin" 
                ;;
            filter)
                opts="--states --coords --output"
                nopts="--help --verbose"
                fopts="--selected-states --every-nth --nRandom" 
                ;;
            stats)
                opts="--states --concat-limits" 
                nopts="--help"
                fopts="--concat-nframes" 
                ;;
            *)
                ;;
        esac
    fi
    # if opts was parsed
    if [[ " ${opts} " == *" ${prev} "* ]]; then
        COMPREPLY=( $(compgen -f -- ${cur}) )
    # if fopts was parse, no completion is returned
    elif [[ " ${fopts} " == *" ${prev} "* ]]; then
        COMPREPLY=()
    # complete one of the given options
    elif [[ "${opts} ${nopts} ${fopts}" == *"${cur}"* ]]; then
        COMPREPLY=( $(compgen -W "${opts} ${nopts} ${fopts}" -- ${cur}) )
    # return fallback file-directory completion
    else
        COMPREPLY=( $(compgen -df -- ${cur}) )    
    fi
	return 0
}
complete -o filenames -F _clustering clustering

