%% Load Data
data = xlsread('Cleaned.xlsx');

feature_names = ["Ground transportation to/from airport","Parking facilities","Parking facilities(value for money)"...
    "Availability of baggage carts","Efficiency of check-in staff","Check-in wait time","Courtesy of of check-in staff"...
    "Wait time at passport inspection",	"Courtesy of inspection staff",	"Courtesy of security staff",...
    "Thoroughness of security inspection",	"Wait time of security inspection", "Feeling of safety and security"...
    "Ease of finding your way through the airport",	"Flight information screens",	"Walking distance inside terminal"...
    "Ease of making connections","Courtesy of airport staff","Restaurants",	"Restaurants (value for money)",...
    "Availability of banks/ATM/money changing",	"Shopping facilities","Shopping facilities (value for money)","Internet access"...
    "Business/executive lounges","Availability of washrooms","Cleanliness of washrooms","Comfort of waiting/gate areas",...
    "Cleanliness of airport terminal", "Ambience of airport", "Arrivals passport and visa inspection", "Speed of baggage delivery",...
    "Customs inspection", "Encoded departure time","Encoded Quarter"];

X = data(:,1:35);
y = data(:, 36);


%% PCA Decomposition

% normalization data mean of 0 and s.d of 1
mu_x = mean(X);
sd_x = std(X);

X_temp = bsxfun(@minus, X, mu_x) ;
X_norm = bsxfun(@rdivide, X_temp, sd_x) ;

n_pcs = 10;
% PCA Implementation
[coeff,score,latent,~,explained] = pca(X_norm, 'NumComponents',n_pcs);

dataset = [score, y];

% 
split_ratio = 0.05;

%% Extracting names of top Principal Components
important = zeros(n_pcs,1);

for i = 1:n_pcs
    [argvalue, argmax] = max(abs(coeff(:,i)));
    important(i) = argmax;
end

important_names = cell(n_pcs,1);

for i = 1:n_pcs
     n = important(i);
     disp(feature_names(1,n))
    important_names{i} = feature_names(1,n) ;
end


%% Support Vector Machine and Decision Trees Implementation
count_val = 0;
count_trn = 0;

svm_count_val = 0;
svm_count_trn = 0;

val_curve = zeros(1,17);
train_curve = zeros(1,17);

svm_train_curve = zeros(1,17);
svm_val_curve = zeros(1,17);

iteration = 1;

for m = 100:100:2200
 
    [X_train, y_train, X_val, y_val] = split_m(dataset,0.1); %split the data
    
    rng(10)
    
    % Training step
    ctree = fitctree(X_train(1:m,:), y_train(1:m),...
     'MinLeafSize',17 );
 
    t = templateSVM('KernelFunction','linear', 'BoxConstraint', 100);
    svm_model = fitcecoc(X_train(1:m,:), y_train(1:m), 'Learners', t);
    
    % Predict training and validation set using Decision Tree
    ypred_val = predict(ctree, X_val); 
    ypred_trn = predict(ctree, X_train(1:m,:));
    
    % Predict training and validation set using SNM
    svm_ypred_val = predict(svm_model, X_val); 
    svm_ypred_trn = predict(svm_model, X_train(1:m,:));
    
    % Plot confusion chart
%     if (m == 100) || (m == 900) || (m == 1700)
%         view(ctree,'mode','graph')
%         figure
%         cm = confusionchart(y_val ,ypred_val);     
%     end
%     
    % Validation misclassification
    for n = 1:size(ypred_val)
        if ypred_val(n) ~= y_val(n)
            count_val = count_val + 1;
        end
        % SVM
         if svm_ypred_val(n) ~= y_val(n)
            svm_count_val = svm_count_val + 1;
         end
    end
   
    count_val;
    
    % Error
    val_curve(iteration) = count_val/length(ypred_val);
    svm_val_curve(iteration) = svm_count_val/length(ypred_val);
    
    % Accuracy
%     val_curve(iteration) = (length(ypred_val) - count_val)/length(ypred_val);
%     svm_val_curve(iteration) = (length(ypred_val) - svm_count_val)/length(ypred_val);
    
    count_val = 0; 
    svm_count_val = 0;
    
    % Trainig misclassification 
    for n = 1:size(y_train(1:m))
        % Decision Tree
         if ypred_trn(n) ~= y_train(n)
            count_trn = count_trn + 1;
         end
         % SVM
         if svm_ypred_trn(n) ~= y_train(n)
            svm_count_trn = svm_count_trn + 1;
         end
    end
          
  
    disp(count_trn)
    % Error 
    train_curve(iteration) = count_trn/length(ypred_trn);
    svm_train_curve(iteration) = svm_count_trn/length(ypred_trn);
    
    % Accuracy
%     train_curve(iteration) = (length(ypred_trn) - count_trn)/length(ypred_trn);
%     svm_train_curve(iteration) = (length(ypred_trn) - svm_count_trn)/length(ypred_trn);
    
    count_trn = 0;
    svm_count_trn = 0;
    
    iteration = iteration + 1;
end
%% PLotting the Learning curve
subplot(1,2,1)
plot(train_curve, 'r')
hold on
plot(val_curve, 'b')
xlabel('iteration');
ylabel('Misclassification errors');
legend('training misclassification','validation misclassification')
title('Decision Tree validation curve')
hold off 



subplot(1,2,2)
plot(svm_train_curve, 'r')
hold on
plot(svm_val_curve, 'b')
xlabel('iteration');
ylabel('errors');
legend('training error','validation error')
title('SVM validation curve')
%% Plot Confusion Matrix


view(ctree,'mode','graph')

figure
cm = confusionchart(y_val ,svm_ypred_val);     
title('SVM confusion matrix')
figure
cm1 = confusionchart(y_val ,ypred_val);    
title('Decision confusion matrix')

%% Decision Tree Hyperparameter optimization
leafs = 1:1:40;
cnt = 0;
n_leafs = length(leafs);
hyp_err = zeros(n_leafs,1);

for n= 1:n_leafs
    
    tree = fitctree(X_train(1:m,:),y_train(1:m), 'MinLeafSize', leafs(n));
    hyp_pred = predict(tree, X_val);
    
    
    for i = 1:size(hyp_pred)
        % Decision Tree
         if hyp_pred(i) ~= y_val(i)
            cnt = cnt + 1;
         end
    end
    
    hyp_err(n) = cnt/length(hyp_pred);
    cnt = 0;
end

%% SVM Hyperparameter optimization
kernels = ["polynomial", "rbf", "linear"];
margin = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100];

cnt = 0;
n_marg = length(margin);
n_kern = length(kernels);
hyp_svm_err = zeros(n_kern,n_marg);

for i= 1:n_marg
    for j = 1:n_kern
      disp(kernels(j))
      disp(margin(i))
      t = templateSVM('KernelFunction',kernels(j), 'BoxConstraint', margin(i) );
      svm = fitcecoc(X_train(1:m,:), y_train(1:m), 'Learners', t);
      
      hyp_svm_pred = predict(svm, X_val);
       for n = 1:size(hyp_svm_pred)
            % Decision Tree
             if hyp_svm_pred(n) ~= y_val(n)
                cnt = cnt + 1;
             end
        end
    
    hyp_svm_err(j,i) = cnt/length(hyp_svm_pred);
    cnt = 0;
    end
end

%% Plotting the Hyperparameter optimization error
figure

subplot(1,2,1)
plot(leafs,hyp_err);
xlabel('Min Leaf Size');
ylabel('cross-validated error');
title('Decision Tree Hyperparameter Optimization')

subplot(1,2,2)
plot(margin,hyp_svm_err)
xlabel('Margin gap (C)');
ylabel('Error');
title('SVM Hyperparameter Optimization')
legend('ploynomial', 'gaussian', 'linear')

