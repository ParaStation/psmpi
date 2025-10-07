use strict;
use MyDef::output_f90;

package MyDef::output_fortran;

our $out;
our $debug;



sub get_interface {
    return (\&init_page, \&parsecode, \&set_output, \&modeswitch, \&dumpout);
}

sub set_output {
    my ($newout)=@_;
    $out = $newout;
    MyDef::output_f90::set_output($newout);
}

sub modeswitch {
    my ($mode, $in)=@_;
}

sub init_page {
    my ($t_page)=@_;
    my $page=$t_page;
    MyDef::set_page_extension("f90");


    my $init_mode = MyDef::output_f90::init_page(@_);
    return $init_mode;
}

sub parsecode {
    my ($l)=@_;
    if ($l=~/^DEBUG (\w+)/) {
        if ($1 eq "OFF") {
            $debug=0;
        }
        else {
            $debug=$1;
        }
        return MyDef::output_f90::parsecode($l);
    }
    elsif ($l=~/^\$eval\s+(\w+)(.*)/) {
        my ($codename, $param)=($1, $2);
        $param=~s/^\s*,\s*//;
        my $t=MyDef::compileutil::eval_sub($codename);
        eval $t;
        if ($@ and !$MyDef::compileutil::eval_sub_error{$codename}) {
            $MyDef::compileutil::eval_sub_error{$codename}=1;
            print "evalsub - $codename\n";
            print "[$t]\n";
            print "eval error: [$@] package [", __PACKAGE__, "]\n";
        }
        return;
    }
    return MyDef::output_f90::parsecode($l);
}

sub dumpout {
    my ($f, $out)=@_;
    MyDef::output_f90::dumpout($f, $out);
}


1;

1;
