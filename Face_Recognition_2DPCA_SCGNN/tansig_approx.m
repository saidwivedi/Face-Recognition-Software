function y = tansig_approx(x)

    if x >= 2
        y = 1;
    elseif (x > -2) && (x < 0)
        y = x*(1+x*0.25);
    elseif (x > 0) && (x < 2)
        y = x*(1-x*0.25);
    else
        y = -1;
    end
