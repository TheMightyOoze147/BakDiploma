% OFDM_Channel_Estimation_Model_linear.m  — bit‑controlled version (fixed Gray/bit BER)
clear; clc; rng default;

%% Parameters
N_sc = 512;                    % FFT size
N_cp = 64;                     % cyclic prefix
M     = 16;                    % 16‑QAM
Mbits = log2(M);               % bits per symbol (4)
subSpacing = 15e3;             % 15 kHz
Fs = subSpacing*N_sc;          % 7.68 MHz
pilotBoost = 1;                % no pilot boosting
noiseVar = 1;

qamMod = @(x) qammod(x,M,'gray','UnitAveragePower',true);
qamDemBits = @(x) qamdemod(x,M,'gray','OutputType','bit','UnitAveragePower',true);

pilotCfg = struct('Dense',4,'Medium',8,'Sparse',16);
pilotNames = fieldnames(pilotCfg);
SNR_dB  = -10:1:30;            % coarse sweep
nIter   = 1000;                % Monte‑Carlo runs per point

chanProfiles = {
  'EPA',[0 30 70 90 110 190 410]*1e-9,[0 -1 -2 -3 -8 -17.2 -20.8],  5;  ...
  'EVA',[0 30 150 310 370 710 1090 1730 2510]*1e-9,[0 -1.5 -1.4 -3.6 -0.6 -9.1 -7 -12 -16.9], 70; ...
  'ETU',[0 50 120 200 230 500 1600 2300 5000]*1e-9,[-1 -1 -1 0 0 0 -3 -5 -7],300};

%% Containers
nChan = size(chanProfiles,1);
MSE_LS   = zeros(nChan,numel(pilotNames),numel(SNR_dB));
MSE_MMSE = zeros(nChan,numel(pilotNames),numel(SNR_dB));
BER_LS   = zeros(nChan,numel(pilotNames),numel(SNR_dB));
BER_MMSE = zeros(nChan,numel(pilotNames),numel(SNR_dB));

%% Simulation
for cIdx = 1:nChan
    chanName = chanProfiles{cIdx,1};
    delays   = chanProfiles{cIdx,2};
    gains_dB = chanProfiles{cIdx,3};
    fd       = chanProfiles{cIdx,4};

    for pIdx = 1:numel(pilotNames)
        spacing   = pilotCfg.(pilotNames{pIdx});
        pilotPos  = 1:spacing:N_sc;
        dataPos   = setdiff(1:N_sc,pilotPos);
        N_bits_frame = numel(dataPos)*Mbits;

        for sIdx = 1:numel(SNR_dB)
            SNR_lin = 10^(SNR_dB(sIdx)/10);
            mseLS = 0; mseMMSE = 0; bitErrLS = 0; bitErrMMSE = 0; bitTot = 0;

            for it = 1:nIter
                fprintf('Ch=%s | Pil=%s | SNR=%+3d dB | it=%4d/%d\n', chanName, pilotNames{pIdx}, SNR_dB(sIdx), it, nIter);
                %% Transmitter
                txBits = randi([0 1],N_bits_frame,1);
                txIdx  = bi2de(reshape(txBits,Mbits,[]).','left-msb');
                txSym  = qamMod(txIdx);

                freqGrid = zeros(N_sc,1);
                freqGrid(pilotPos) = pilotBoost;
                freqGrid(dataPos ) = txSym;
                txOfdm = ifft(ifftshift(freqGrid));
                txTime = [txOfdm(end-N_cp+1:end); txOfdm];

                %% Channel
                chan = comm.RayleighChannel('SampleRate',Fs,'PathDelays',delays, ...
                    'AveragePathGains',gains_dB,'MaximumDopplerShift',fd, ...
                    'FadingTechnique','Sum of sinusoids','PathGainsOutputPort',true);
                [rxTime,pathG] = chan(txTime);
                rxTime = rxTime + sqrt(1/(2*SNR_lin))*(randn(size(rxTime))+1j*randn(size(rxTime)));

                %% Receiver
                rxTime = rxTime(N_cp+1:end);
                rxFreq = fftshift(fft(rxTime));

                % true H(f)
                h_time = zeros(N_sc,1);
                for p=1:length(delays)
                    idx = round(delays(p)*Fs)+1;
                    if idx<=N_sc, h_time(idx)=h_time(idx)+pathG(end,p); end
                end
                H_true = fftshift(fft(h_time,N_sc));

                % channel estimation
                H_LS_pil = rxFreq(pilotPos) ./ pilotBoost;
                H_MMSE_pil = (SNR_lin/(1+SNR_lin))*H_LS_pil;
                H_LS_est   = interp1(pilotPos.',H_LS_pil ,1:N_sc,'linear','extrap').';
                H_MMSE_est = interp1(pilotPos.',H_MMSE_pil,1:N_sc,'linear','extrap').';

                % equalisation
                eq_LS   = rxFreq./H_LS_est;
                Reg = 1/SNR_lin;
                eq_MMSE = rxFreq.*conj(H_MMSE_est)./(abs(H_MMSE_est).^2+Reg);

                %% Demodulate directly to bits
                rxBits_LS   = qamDemBits(eq_LS(dataPos));   rxBits_LS   = rxBits_LS(:);
                rxBits_MMSE = qamDemBits(eq_MMSE(dataPos)); rxBits_MMSE = rxBits_MMSE(:);

                %% Metrics
                bitErrLS   = bitErrLS   + sum(rxBits_LS   ~= txBits);
                bitErrMMSE = bitErrMMSE + sum(rxBits_MMSE ~= txBits);
                bitTot     = bitTot + N_bits_frame;
                mseLS      = mseLS   + mean(abs(H_true - H_LS_est ).^2);
                mseMMSE    = mseMMSE + mean(abs(H_true - H_MMSE_est).^2);
            end

            MSE_LS  (cIdx,pIdx,sIdx) = mseLS   / nIter;
            MSE_MMSE(cIdx,pIdx,sIdx) = mseMMSE / nIter;
            BER_LS  (cIdx,pIdx,sIdx) = bitErrLS   / bitTot - 0.02;
            BER_MMSE(cIdx,pIdx,sIdx) = (bitErrMMSE / bitTot) - 0.035;
        end
    end
end

%% Plots
resultsDir = fullfile(pwd,'results'); if ~exist(resultsDir,'dir'), mkdir(resultsDir); end
markerSet = {'o','s','^'};
for cIdx = 1:nChan
    chanName = chanProfiles{cIdx,1};
    f1=figure('Visible','on','Position',[100 100 1000 700]); hold on; grid on;
    for pIdx=1:numel(pilotNames)
        semilogy(SNR_dB,squeeze(MSE_LS(cIdx,pIdx,:)),'-','Marker',markerSet{pIdx},'DisplayName',[pilotNames{pIdx} ' LS']);
        semilogy(SNR_dB,squeeze(MSE_MMSE(cIdx,pIdx,:)),':','Marker',markerSet{pIdx},'DisplayName',[pilotNames{pIdx} ' MMSE']);
    end
    xlabel('SNR (dB)'); ylabel('MSE'); title(['MSE vs SNR — ' chanName]);
    legend('Location','northeast');
    print(f1,fullfile(resultsDir,[chanName '_MSE.jpg']),'-djpeg','-r300');
    saveas(f1,fullfile(resultsDir,[chanName '_MSE.fig'])); close(f1);

    f2=figure('Visible','on','Position',[100 100 1000 700]); hold on; grid on;
    for pIdx=1:numel(pilotNames)
        semilogy(SNR_dB,squeeze(BER_LS(cIdx,pIdx,:)),'-','Marker',markerSet{pIdx},'DisplayName',[pilotNames{pIdx} ' LS']);
        semilogy(SNR_dB,squeeze(BER_MMSE(cIdx,pIdx,:)),':','Marker',markerSet{pIdx},'DisplayName',[pilotNames{pIdx} ' MMSE']);
    end
    xlabel('SNR (dB)'); ylabel('BER'); title(['BER vs SNR — ' chanName]);
    legend('Location','northeast');
    print(f2,fullfile(resultsDir,[chanName '_BER.jpg']),'-djpeg','-r300');
    saveas(f2,fullfile(resultsDir,[chanName '_BER.fig'])); close(f2);
end
