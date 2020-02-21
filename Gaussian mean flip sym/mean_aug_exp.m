%% Set parameters

n = 100;
n_mc = 1e2;
rng(2);
MSE_mc = zeros(n_mc,1);
MSE_aug_mc = zeros(n_mc,1);

for j=1:n_mc
    %% MC simu
    theta_true = 0;
    X = theta_true+randn(n,1);
    X_aug = (X+ flipud(X))/2;
    
    MSE_mc(j) = norm(X-theta_true)^2/n;
    MSE_aug_mc(j) = norm(X_aug-theta_true)^2/n;
    
end

    MSE  = mean(MSE_mc);
    MSE_aug = mean(MSE_aug_mc);

%%
figure, hold on
rng(2); a = {'-','--','-.',':'};
h1 = histogram(MSE_mc);
set(h1,'LineStyle',a{1});
h2 = histogram(MSE_aug_mc);
set(h2,'LineStyle',a{2});
xlabel('MSE')
ylabel('count')
%ylim([0,max(max(MSE),max(MSE_aug))])
set(gca,'fontsize',20)


legend([h1,h2],{'MLE','Aug MLE'},'location','Best')

str = sprintf( 'd=%d, n_{mc} = %d',n,n_mc);
title(str);

savefigs = 1;
if savefigs==1
    filename = ...
        sprintf( './aug-mean-estim-flip-n=%d-n-mc=%d.png',...
        n,n_mc);
    saveas(gcf, filename,'png');
    fprintf(['Saved Results to ' filename '\n']);
end