%% Set parameters

d_arr =(1:5:100);
L = length(d_arr);
RE= zeros(L,1);

for i=1:L
    
    d = d_arr(i);
    n_mc = 1e2;
    rng(2);
    MSE_mc = zeros(n_mc,1);
    MSE_aug_mc = zeros(n_mc,1);
    
    for j=1:n_mc
        %% MC simu
        
        X = randn(d,1);
        v = X';
        
        C_X = toeplitz([v(1) fliplr(v(2:end))], v);
        
        
        S = X*X';
        MSE_mc(j) = trace(S^2);
        S = C_X*C_X'/d;
        MSE_aug_mc(j) = trace(S^2);
        
    end
    
    MSE  = mean(MSE_mc);
    MSE_aug = mean(MSE_aug_mc);
    RE(i) = MSE/MSE_aug;
    
end
%%
figure, hold on
rng(2); a = {'-','--','-.',':'};
h1 = plot(d_arr,RE,'linewidth',3,'color',rand(1,3));
set(h1,'LineStyle',a{1});
xlabel('dimension d')
ylabel('RE')
%ylim([0,max(max(MSE),max(MSE_aug))])
set(gca,'fontsize',20)


%legend([h1,h2],{'MLE','Aug MLE'},'location','Best')

%str = sprintf( 'n=%d, n_{mc} = %d',n,n_mc);
%title(str);

savefigs = 1;
if savefigs==1
    filename = ...
        sprintf( './aug-2L-NN-circ-shift-n-mc=%d.png',...
       n_mc);
    saveas(gcf, filename,'png');
    fprintf(['Saved Results to ' filename '\n']);
end