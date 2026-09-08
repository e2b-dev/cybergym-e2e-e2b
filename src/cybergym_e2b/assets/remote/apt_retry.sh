#!/bin/sh
set -eu

real_dir=${E2B_APT_REAL_DIR:-/usr/bin}
real_command="$real_dir/$(basename "$0")"

case " $* " in
  *" update "*)
    attempt=1
    while [ "$attempt" -le 4 ]; do
      if "$real_command" \
        -o Acquire::Retries=3 \
        -o Acquire::http::No-Cache=true \
        "$@"
      then
        exit 0
      fi
      find /var/lib/apt/lists/partial -mindepth 1 -maxdepth 1 -type f -delete \
        2>/dev/null || true
      attempt=$((attempt + 1))
      if [ "$attempt" -le 4 ]; then
        sleep "${E2B_APT_RETRY_SLEEP:-1}"
      fi
    done
    exit 1
    ;;
esac

exec "$real_command" "$@"
