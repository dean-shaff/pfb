function out=polyphase_analysis_padded(...
    in,...
    filt,...
    block,...
    os_factor,...
    verbose_...
)

  % Args:
  %   in ([numeric]): input data. The dimensionality should be
  %     (n_pol, n_chan, n_dat), where n_chan is equal to 1.
  %   filt ([numeric]): prototype lowpass filter
  %     (length should be multiple of step). Should be single dimensional array.
  %   block (numeric): length of fft
  %     (prefilter length = length(filt)/block
  %     if not the 'filt' is padded with zeros to a multiple of block
  %     Importantly, This is also the number of channels that will
  %     be created by the PFB.
  %   os_factor (struct): rational number struct; struct with 'nu' and 'de' fields
  %   verbose_ (bool): Optional. verbosity flag. Defaults to false.
  % Return:
  %   [numeric]: The first dimension is time, the second frequency.
  %     The number of frequency channels is equal to ``block``


  %The Custom MATLAB function
  % y_Nc = OS_Poly_DFT_FB_Batch(x)
  %performs the Over-Sampled Channelization of the input vector 'in' and
  %yeilds the time series 'out' for all channels'. There are configuration
  %parameters passed to the function via the 'OS_Poly_DFT_FB_Batch_Config_para.mat'.
  % Nc => Number of Channels
  % Num => Numerator and
  % Den => Deniminato of the Over-Sampling-Factor Os = Num/Den
  % h2D => [Nc,Nt] size 2D Matrix containing the Segmented filter
  %        coefficients where Nc is the number of channels and Nt is the
  %        maximum number of tap per polyphase arm.
  % BRI => Barrel Rotator Index. Adjust this to avoid phase offsets.
  %
  % By Thushara Gunaratne - RO/RCO - NSI-NRC CANADA
  % Start Date - 2017-07-10
  tstart = tic;

  verbose = 0;

  if exist('verbose_', 'var')
    verbose = verbose_;
  end

  in_size = size(in);
  n_pol = in_size(1);
  n_chan = in_size(2); % This should always be 1.
  n_dat = in_size(3);
  dtype = class(in);
  is_real = isreal(in);

  filt = cast(filt, dtype);

  %The Commutator Length
  % John Bunton calls this `step`
  % John Bunton calls Nc `block`
  step = floor((block * os_factor.de) / os_factor.nu);
  overlap = block - step;


  % CL = Nc*Den/Num; % if Nc == 512 and os==8/7 then CL = 448
  % L_M = Nc - CL; % if Nc == 512 and os=8/7 then L_M = 64

  % The Number of Output Samples
  % in John's code, this is `nblocks`
  % where `nblocks=floor((Ni - length(filter))/CL);`
  % say, for example, that Ni is 344064 (taken from a real example) and we're
  % using a filter of length 12545 that has been zero padded to be of length 12800.
  % This code will result in No = 768, while Johns code will result in No = 739

  nblocks = ceil(n_dat / step);
  % No = ceil(Ni/CL);

  % in John's code, Nt is `phases`, which is equal to ceil(length(filter)/nchan);
  % if length(filter) == 12545 and nchan == 512, then Nt = 25

  %The 2D Data-Mask
  % [~,Nt] = size(h2D);
  % xM2D = zeros(Nc,Nt);
  % %The 1D data-Mask
  % xM = reshape(xM2D,Nc*Nt,1);
  %
  %Initiating the Output Vector for the Selected-Channel
  % y_Nc = zeros(No,Nc);
  sample_delay_shift = ceil((length(filt)-1)/2/step);
  % sample_delay_shift = ceil((length(filt)-1)/2/block);

  if verbose
    fprintf('polyphase_analysis_padded: sample_delay_shift=%d\n', sample_delay_shift);
  end


  filt_padded = pad_filter(filt, block);
  phases = length(filt_padded) / block;
  filt_padded_2d = reshape(filt_padded, block, phases);

  in_mask_2d = zeros(block, phases);
  in_mask = reshape(in_mask_2d, block*phases, 1);

  out = zeros(n_pol, block, nblocks);

  for i_pol = 1:n_pol
    if verbose
      fprintf('polyphase_analysis_padded: %d/%d pol\n', i_pol, n_pol);
    end
    in_pol = squeeze(in(i_pol, 1, :));
    %Iterative Filtering
    BRI = 0;
    for idx = 1:nblocks
        % if verbose
        %   fprintf('polyphase_analysis_padded: %d/%d blocks\n', idx, nblocks);
        % end
        %% Polyphase Filtering
        yPFB = sum(filt_padded_2d.*in_mask_2d, 2);
        %Updating the Input Vector
        %Shift the Current Samples by M to the Right
        in_mask(step+1:end) = in_mask(1:end-step);
        %Assign the New Input Samples for the first M samples
        in_mask(1:step) = in_pol(idx*step:-1:(idx-1)*step+1);%Note the Flip (Left-Right) place the Newest sample to the front
        %ReShaping in_mask
        in_mask_2d = reshape(in_mask, block, phases);

        %% Taking the Nc-Point IFFT

        %Performing the Circular Shift to Compensate the Shift in Band Center
        %Frequencies
        if BRI == 0
            y1S = yPFB;
        else
            % y1S = [yPFB(mod((os_factor.nu-BRI)*overlap+1, block):end); ...
            %        yPFB(1:mod((os_factor.nu-BRI)*overlap, block))];
            index = mod((os_factor.nu-BRI)*overlap, block);
            % index_john = (step*(idx-1) - floor(step*(idx-1)/block)*block);
            % if index ~= index_john
            %   fprintf('ERROR!!!!\n');
            % end
            % fprintf('index=%d, index_john=%d\n', index, index_john);
            y1S = circshift(yPFB, -index);
        end

        %Evaluating the Cross-Stream (i.e. column wise) IDFT
        out(i_pol, :, idx) = (block^2)*ifft(y1S);
        % out(i_pol, :, idx) = block*fft(y1S);

        %Updating the Barrel-Roter Index
        BRI = mod(BRI+1, os_factor.nu);

    end
  end

  out = circshift(out, [0, 0, -sample_delay_shift]);
  if verbose
    tdelta = toc(tstart);
    fprintf('polyphase_analysis_padded: Elapsed time is %f seconds\n', tdelta);
  end
end
