#!/bin/bash
#
# Usage:
#
# - Show hashes:   > ./list-of-patches.sh
# - Add new patch: > git show `./list-of-patches.sh <hash of new patch>` > list-of-patches.txt
# - Rebuild list:  > ./list-of-patches.sh `grep "commit " list-of-patches.txt | cut -d" " -f2`
#                  > git show `./list-of-patches.sh` > list-of-patches.txt
#
patchfiles=`ls 0*.patch 2> /dev/null`
commits=""
number=0
for file in $patchfiles ; do
	hash=`head -1 "$file"  | cut -d" " -f2`
	commits="$commits $hash"
	#git --no-pager show $hash
	number=$[$number + 1]
done
while [ $# -gt 0 ] ; do
	commits="$commits $1"
	number=$[$number + 1]
	git format-patch --start-number $number --no-signature --quiet -1 $1 
	shift
done
echo $commits

