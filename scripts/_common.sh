# Shared helpers, sourced by the launchers in this directory. Not meant to be run directly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

# Log to wandb only if a key is present in the environment; otherwise log offline.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
	export WANDB_MODE="offline"
fi

# Pick a free master port for deepspeed.
pick_master_port() {
	comm -23 \
		<(seq 10000 65535 | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
}

# Prepare OUTPUT_DIR (absolute, git-ignored, with a copy of the launcher and tee'd logs).
prepare_output_dir() {
	mkdir -p "${OUTPUT_DIR}"
	OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
	[[ -f "${OUTPUT_DIR}/.gitignore" ]] || echo '*' >"${OUTPUT_DIR}/.gitignore"
	cp -f "$0" "${OUTPUT_DIR}/script.sh"
	exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)
}

# Parse `--flag value` / `--flag=value` pairs into the variables named by upper-casing the flag.
# Usage: parse_flags "$@"   (unknown flags abort).
parse_flags() {
	while [[ "$#" -gt 0 ]]; do
		local arg="$1"
		shift
		case "${arg}" in
			--*=*) local key="${arg%%=*}" val="${arg#*=}" ;;
			--*) local key="${arg}" val="$1"; shift ;;
			*) echo "Unexpected argument: ${arg}" >&2; exit 1 ;;
		esac
		key="$(echo "${key#--}" | tr '[:lower:]' '[:upper:]')"
		if ! declare -p "${key}" &>/dev/null; then
			echo "Unknown flag: --${key,,}" >&2
			exit 1
		fi
		printf -v "${key}" '%s' "${val}"
	done
}
