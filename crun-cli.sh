#!/bin/sh
# Safe llama-cli wrapper for Android (HTP / Adreno) with proper argument escaping

# ---- CONFIG ----
basedir=/data/local/tmp/llama.cpp

branch="."
[ -n "$B" ] && branch="$B"

adbserial=""
[ -n "$S" ] && adbserial="-s $S"

model="Llama-3.2-3B-Instruct-Q4_0.gguf"
[ -n "$M" ] && model="$M"

device="HTP0"
[ -n "$D" ] && device="$D"

verbose=""
[ -n "$V" ] && verbose="GGML_HEXAGON_VERBOSE=$V"

experimental=""
[ -n "$E" ] && experimental="GGML_HEXAGON_EXPERIMENTAL=$E"

sched=""
if [ -n "$SCHED" ]; then
    sched="GGML_SCHED_DEBUG=2"
    cli_opts="$cli_opts -v"
fi

profile=""
[ -n "$PROF" ] && profile="GGML_HEXAGON_PROFILE=$PROF GGML_HEXAGON_OPSYNC=1"

opmask=""
[ -n "$OPMASK" ] && opmask="GGML_HEXAGON_OPMASK=$OPMASK"

nhvx=""
[ -n "$NHVX" ] && nhvx="GGML_HEXAGON_NHVX=$NHVX"

ndev=""
[ -n "$NDEV" ] && ndev="GGML_HEXAGON_NDEV=$NDEV"

# -------------------------
# SAFE ARG ESCAPING SECTION
# -------------------------
escape() {
    # Escape double quotes AND backslashes for adb shell
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

args=""
for a in "$@"; do
    esc=$(escape "$a")
    args="$args \"$esc\""
done

set -x

# -------------------------
# RUN ON DEVICE
# -------------------------
adb $adbserial shell " \
  cd $basedir; ulimit -c unlimited; \
  LD_LIBRARY_PATH=$basedir/$branch/lib \
  ADSP_LIBRARY_PATH=$basedir/$branch/lib \
  $verbose $experimental $sched $opmask $profile $nhvx $ndev \
    ./$branch/bin/llama-cli --no-mmap \
      -m $basedir/../gguf/$model \
      --poll 0 -t 6 --cpu-mask 0xfc --cpu-strict 1 \
      --ctx-size 8192 --batch-size 128 \
      -ctk q8_0 -ctv q8_0 -fa on \
      -ngl 99 --device $device \
      $cli_opts $args \
"

