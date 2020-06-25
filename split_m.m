
function [X_train, y_train, X_val, y_val] = split_m(data, split_ratio)
    rand_idx = randperm(length(data(:,1)));
    new_data = data(rand_idx,:);
    cut = length(data(:,1))*split_ratio;
    disp(cut)
    n = length(data(1,:)) - 1 ;
    X_train = new_data(cut:end, 1:n);
    y_train = new_data(cut:end, end);
    X_val = new_data(1:cut, 1:n);
    y_val = new_data(1:cut, end);
end