bin=$HOME/bin
lib=$HOME/lib/perl5
def=$HOME/lib/MyDef

mkdir -p bin lib/perl5/MyDef lib/MyDef

A_set="make page run"
for a in $A_set ; do
    echo "    mydef_$a"
    cp -f $bin/mydef_$a bin/
done

echo ""
echo "    MyDef.pm"
cp $lib/MyDef.pm lib/perl5/

A_set="parseutil compileutil dumpout utils regex ext"
for a in $A_set ; do
    echo "    MyDef/$a.pm"
    cp $lib/MyDef/$a.pm  lib/perl5/MyDef/
done

# ---- modules ----
echo ""
A_set="general perl c cpp java fortran"
for a in $A_set ; do
    echo "    output_$a"
    cp $lib/MyDef/output_$a.pm  lib/perl5/MyDef/
    cp $def/std_$a.def lib/MyDef/
done
