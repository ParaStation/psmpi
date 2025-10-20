use strict;
package MyDef::ext;

# ---- subroutines --------------------------------------------
sub grab_codelist {
    my (%opt) = @_;
    my $codelist = $MyDef::compileutil::named_blocks{"last_grab"};
    my $do_macro = $opt{do_macro};
    my $with_indent = $opt{with_indent};
    if ($codelist) {
        my (@t, $indent);
        foreach my $t (@$codelist) {
            if ($do_macro) {
                MyDef::compileutil::expand_macro(\$t);
            }

            if ($t=~/^SOURCE:/) {
            }
            elsif ($t=~/^NEWLINE/) {
                push @t, "\n";
            }
            else {
                if ($with_indent) {
                    push @t, $t;
                }
                else {
                    if ($t=~/^SOURCE_INDENT/) {
                        $indent++;
                    }
                    elsif ($t=~/^SOURCE_DEDENT/) {
                        $indent--;
                    }
                    elsif ($indent>0) {
                        push @t, "    "x$indent . $t;
                    }
                    else {
                        push @t, $t;
                    }
                }
            }
        }

        while ($t[-1]=~/^\s*$/) {
            pop @t;
        }
        $codelist = \@t;
    }
    return $codelist;
}

sub inject_sub {
    my ($name, $src) = @_;
    my $param;
    if ($name=~/(\w+)\s*(\(.*\))/) {
        ($name, $param)=($1, $2);
    }
    my $t_code=MyDef::parseutil::new_code("sub", $name, 9, $param);
    $t_code->{source}=$src;
    $MyDef::def->{codes}->{$name}=$t_code;
}

sub run_src {
    my ($src) = @_;
    my $t_code=MyDef::parseutil::new_code("sub", "_", 9);
    $t_code->{source}=$src;
    $MyDef::def->{codes}->{"_"}=$t_code;
    MyDef::compileutil::call_sub("_");
}

sub grab_ogdl {
    my ($is_list) = @_;
    my $codelist = grab_codelist("do_macro"=>1, "with_indent"=>1);
    if ($codelist) {
        my $ogdl;
        if ($is_list) {
            $ogdl = [];
        }
        else {
            $ogdl = {};
        }
        my @stack;
        my $cur=$ogdl;
        my $last_key;
        foreach my $t (@$codelist) {
            if ($t=~/^SOURCE_INDENT/) {
                if ($last_key) {
                    my $t = {"_"=>$cur->{$last_key}};
                    $cur->{$last_key} = $t;
                    push @stack, $cur;
                    $cur = $t;
                }
                else {
                    my $tmp = pop @$cur;
                    my $t = {"_"=>$tmp};
                    push @$cur, $t;
                    push @stack, $cur;
                    $cur = $t;
                }
                undef $last_key;
            }
            elsif ($t=~/^SOURCE_DEDENT/) {
                if (@stack) {
                    $cur = pop @stack;
                }
                else {
                    die "grab_ogdl: assert\n";
                }
            }
            elsif ($t=~/^\s*$/) {
                next;
            }
            elsif (!@stack and $is_list) {
                push @$cur, $t;
            }
            elsif ($t=~/^(\w+):\s*(.*)/) {
                $cur->{$1} = $2;
                $last_key = $1;
            }
            else {
                warn "grab_ogdl: error in [$t]\n";
                return undef;
            }
        }
        return $ogdl;
    }
    else {
        return undef;
    }
}

sub grab_file {
    my ($file, $pat) = @_;
    my @t;
    if ($file eq "-") {
        $file = $MyDef::def->{file};
    }
    my $fname=$file;
    if ($file=~/def\/(.*)/) {
        $fname = $1;
    }
    elsif ($file=~/.*\/(.*)/) {
        $fname = $1;
    }

    if ($pat) {
        push @t, "#---- $fname: $pat ----\n";
    }
    else {
        push @t, "#---- file: $fname ----\n";
    }

    my $flag;
    if (open In, $file) {
        while(<In>){
            if ($pat) {
                if (/^\#----\s*$pat\s*----/) {
                    $flag=1;
                }
                elsif (/^\#----.*----/) {
                    $flag=0;
                }
                elsif (!$flag and /^\s*#\s*--\s*$pat\s*--/) {
                    $flag=2;
                }
                elsif ($flag==2 and /^\s*#\s*--.*--/) {
                    $flag=0;
                }
                elsif ($flag) {
                    push @t, $_;
                }
            }
            else {
                push @t, $_;
            }
        }
        close In;
    }
    else {
        die "Can't open $file\n";
    }
    while ($t[-1]=~/^\s*$/) {
        pop @t;
    }

    return \@t;

}

1;
