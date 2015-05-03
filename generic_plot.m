clear
filename = 'cpu_results';
designator = '';
plot_title = 'CPU Execution Time';
plot_x = 'Hashes';
plot_y = 'Time (ns)';

%Parse Data
data = csvread(filename);
x = data(:,1);
y = data(:,2);
[r,c] = size(y);

%Custom Pre-processing

%Plotting
h = figure;
plot(x,y);
title(plot_title);
xlabel(plot_x);
ylabel(plot_y);

print(h,'-dpng', [filename,designator,'.png']);

%Custom Post-Processing

%Calculate CPEs assuming x=hashes, y=execution time
cpe = 1:c;
for i=1:c
	a=polyfit(x,y(:,i),1); %Compute linear aproximation
	cpe(i)=a(1);
end
%hashrate
1/(cpe(1)*1e-9)
%csvwrite([filename,designator,'.csv'],cpe);
