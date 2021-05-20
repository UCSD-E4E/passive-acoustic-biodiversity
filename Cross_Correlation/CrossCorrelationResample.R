cross_correlation2<-function (X = NULL, wl = 512, bp = "pairwise.freq.range", ovlp = 70, 
          dens = NULL, wn = "hanning", cor.method = "pearson", parallel = 1, 
          path = NULL, pb = TRUE, na.rm = FALSE, cor.mat = NULL, output = "cor.mat", 
          compare.matrix = NULL, type = "spectrogram", nbands = 40, 
          method = 2,step=1) 
{
  on.exit(pbapply::pboptions(type = .Options$pboptions$type), 
          add = TRUE)
  argms <- methods::formalArgs(cross_correlation)
  opt.argms <- if (!is.null(getOption("warbleR"))) 
    getOption("warbleR")
  else SILLYNAME <- 0
  names(opt.argms)[names(opt.argms) == "wav.path"] <- "path"
  opt.argms <- opt.argms[!sapply(opt.argms, is.null) & names(opt.argms) %in% 
                           argms]
  call.argms <- as.list(base::match.call())[-1]
  opt.argms <- opt.argms[!names(opt.argms) %in% names(call.argms)]
  if (length(opt.argms) > 0) 
    for (q in 1:length(opt.argms)) assign(names(opt.argms)[q], 
                                          opt.argms[[q]])
  if (is.null(path)) 
    path <- getwd()
  else if (!dir.exists(path) & !is_extended_selection_table(X)) 
    stop("'path' provided does not exist")
  else path <- normalizePath(path)
  if (!any(is.data.frame(X), is_selection_table(X), is_extended_selection_table(X))) 
    stop("X is not of a class 'data.frame', 'selection_table' or 'extended_selection_table'")
  if (is_extended_selection_table(X) & length(unique(attr(X, 
                                                          "check.results")$sample.rate)) > 1) 
    stop("all wave objects in the extended selection table must have the same sampling rate (they can be homogenized using resample_est_waves())")
  if (any(is.na(c(X$end, X$start)))) 
    stop("NAs found in start and/or end")
  if (nrow(X) == 1 & is.null(compare.matrix)) 
    stop("you need more than one selection to do cross-correlation")
  if (bp[1] == "pairwise.freq.range" & is.null(X$bottom.freq)) 
    stop("'bp' must be supplied when no frequency range columns are found in 'X' (bottom.freq & top.freq)")
  if (is.null(bp[1])) 
    stop("'bp' must be supplied")
  if (!is.null(dens)) 
    write(file = "", x = "'dens' has been deprecated and will be ignored")
  if (!is.null(cor.mat)) 
    write(file = "", x = "'dens' has been deprecated and will be ignored")
  if (!any(output %in% c("cor.mat", "list"))) 
    stop("'output' must be either 'cor.mat' or 'list'")
  if (!is.numeric(wl)) 
    stop("'wl' must be a numeric vector of length 1")
  else {
    if (!is.vector(wl)) 
      stop("'wl' must be a numeric vector of length 1")
    else {
      if (!length(wl) == 1) 
        stop("'wl' must be a numeric vector of length 1")
    }
  }
  if (!is.numeric(ovlp)) 
    stop("'ovlp' must be a numeric vector of length 1")
  else {
    if (!is.vector(ovlp)) 
      stop("'ovlp' must be a numeric vector of length 1")
    else {
      if (!length(ovlp) == 1) 
        stop("'ovlp' must be a numeric vector of length 1")
    }
  }
  if (!is_extended_selection_table(X)) {
    fs <- list.files(path = path, pattern = "\\.wav$", ignore.case = TRUE)
    if (length(unique(X$sound.files[(X$sound.files %in% 
                                     fs)])) != length(unique(X$sound.files))) 
      write(file = "", x = paste(length(unique(X$sound.files)) - 
                                   length(unique(X$sound.files[(X$sound.files %in% 
                                                                  fs)])), ".wav file(s) not found"))
    d <- which(X$sound.files %in% fs)
    if (length(d) == 0) {
      stop("The .wav files are not in the working directory")
    }
    else {
      X <- X[d, ]
    }
  }
  if (!is.numeric(parallel)) 
    stop("'parallel' must be a numeric vector of length 1")
  if (any(!(parallel%%1 == 0), parallel < 1)) 
    stop("'parallel' should be a positive integer")
  if (is_extended_selection_table(X) & length(unique(attr(X, 
                                                          "check.results")$sample.rate)) > 1) 
    stop("sampling rate must be the same for all selections")
  X$selection.id <- paste(X$sound.files, X$selec, sep = "-")
  if (!is.null(compare.matrix)) {
    X <- X[X$selection.id %in% unique(c(compare.matrix)), 
           , drop = FALSE]
    if (!any(names(call.argms) == "method")) 
      method <- 2
  }
  if (pb) {
    max.stps <- getOption("warbleR.steps")
    if (is.null(max.stps)) 
      if (method == 1) 
        max.stps <- 2
    else max.stps <- 1
  }
  if (is.null(compare.matrix)) 
    spc.cmbs.org <- spc.cmbs <- t(combn(X$selection.id, 
                                        2))
  else {
    if (all(c(compare.matrix) %in% X$selection.id)) 
      spc.cmbs.org <- spc.cmbs <- compare.matrix
    else {
      complt.sf <- setdiff(c(compare.matrix), X$selection.id)
      wvdr <- duration_wavs(files = complt.sf, path = path)
      names(wvdr)[2] <- "end"
      wvdr$start <- 0
      wvdr$selec <- "whole.file"
      wvdr$selection.id <- paste(wvdr$sound.files, wvdr$selec, 
                                 sep = "-")
      out <- lapply(1:nrow(wvdr), function(x) {
        sls <- setdiff(c(compare.matrix[compare.matrix[, 
                                                       1] %in% wvdr$sound.files[x] | compare.matrix[, 
                                                                                                    2] %in% wvdr$sound.files[x], ]), wvdr$sound.files[x])
        suppressWarnings(df <- data.frame(wvdr[x, ], 
                                          channel = min(X$channel[X$selection.id %in% 
                                                                    sls]), bottom.freq = min(X$bottom.freq[X$selection.id %in% 
                                                                                                             sls]), top.freq = max(X$top.freq[X$selection.id %in% 
                                                                                                                                                sls])))
        return(df)
      })
      wvdr <- do.call(rbind, out)
      int.nms <- intersect(names(X), names(wvdr))
      X <- rbind(as.data.frame(X)[, int.nms, drop = FALSE], 
                 wvdr[, int.nms])
      for (i in complt.sf) compare.matrix[compare.matrix == 
                                            i] <- paste(i, "whole.file", sep = "-")
      spc.cmbs.org <- spc.cmbs <- compare.matrix
    }
  }
  spc_FUN <- function(clp, wlg, ovl, w, nbnds) {
    if (type == "spectrogram") 
      spc <- seewave::spectro(wave = clp, wl = wlg, ovlp = ovl, 
                              wn = w, plot = FALSE, fftw = TRUE, norm = TRUE)
    if (type == "mfcc") {
      spc <- melfcc(clp, nbands = nbnds, hoptime = (wlg/clp@samp.rate) * 
                      (ovl/100), wintime = wlg/clp@samp.rate, dcttype = "t3", 
                    fbtype = "htkmel", spec_out = TRUE)
      names(spc)[2] <- "amp"
      spc$amp <- t(spc$amp)
      spc$freq <- seq(0, clp@samp.rate/2000, length.out = nbnds)
    }
    spc$amp[is.infinite(spc$amp)] <- NA
    return(spc)
  }
  if (method == 1) {
    if (pb) 
      if (type == "spectrogram") 
        write(file = "", x = paste0("creating spectrogram matrices (step 1 of ", 
                                    max.stps, "):"))
    else write(file = "", x = paste0("creating MFCC matrices (step 1 of ", 
                                     max.stps, "):"))
    pbapply::pboptions(type = ifelse(pb, "timer", "none"))
    if (Sys.info()[1] == "Windows" & parallel > 1) 
      cl <- parallel::makePSOCKcluster(getOption("cl.cores", 
                                                 parallel))
    else cl <- parallel
    spcs <- pbapply::pblapply(X = 1:nrow(X), cl = cl, function(e){
      clp <- warbleR::read_wave(X = X, index = e, path = path)
      spc_FUN(clp, wlg = wl, ovl = ovlp, w = wn, nbnds = nbands)})
    if (!is_extended_selection_table(X) & length(unique(sapply(spcs, 
                                                               function(x) length(x$freq)))) > 1) 
      stop("sampling rate must be the same for all selections")
    names(spcs) <- X$selection.id
  }
  XC_FUN <- function(spc1, spc2, b = bp, cm = cor.method,step=step) {
    spc1$amp <- spc1$amp[spc1$freq >= b[1] & spc1$freq <= 
                           b[2], ]
    spc2$amp <- spc2$amp[which(spc2$freq >= b[1] & spc2$freq <= 
                                 b[2]), ]
    if (ncol(spc1$amp) > ncol(spc2$amp)) {
      lg.spc <- spc1$amp
      shrt.spc <- spc2$amp
    }
    else {
      lg.spc <- spc2$amp
      shrt.spc <- spc1$amp
    }
    shrt.lgth <- ncol(shrt.spc) - 1
    stps <- ncol(lg.spc) - ncol(shrt.spc)
    if (stps <= 1) 
      stps <- 1
    else stps <- seq(1,stps,step)
    cors <- sapply(stps, function(x, cor.method = cm) {
      warbleR::try_na(cor(c(lg.spc[, x:(x + shrt.lgth)]), 
                          c(shrt.spc), method = cm, use = "pairwise.complete.obs"))
    })
    return(cors)
  }
  ord.shuf <- sample(1:nrow(spc.cmbs))
  spc.cmbs <- spc.cmbs[ord.shuf, , drop = FALSE]
  if (pb) 
    write(file = "", x = paste0("running cross-correlation (step ", 
                                max.stps, " of ", max.stps, "):"))
  if (Sys.info()[1] == "Windows" & parallel > 1) 
    cl <- parallel::makePSOCKcluster(getOption("cl.cores", 
                                               parallel))
  else cl <- parallel
  xcrrs <- pbapply::pblapply(X = 1:nrow(spc.cmbs), cl = cl, 
                             FUN = function(j, BP = bp, cor.meth = cor.method) {
                               if (BP[1] %in% c("pairwise.freq.range", "frange")) 
                                 BP <- c(min(X$bottom.freq[X$selection.id %in% 
                                                             spc.cmbs[j, ]]), max(X$top.freq[X$selection.id %in% 
                                                                                               spc.cmbs[j, ]]))
                               if (method == 1) {
                                 spc1 <- spcs[[spc.cmbs[j, 1]]]
                                 spc2 <- spcs[[spc.cmbs[j, 2]]]
                               }
                               if (method == 2) {
                                 clp1 <- warbleR::read_wave(X = X, index = which(X$selection.id == 
                                                                                  spc.cmbs[j, 1]), path = path)
                                 clp2 <- warbleR::read_wave(X = X, index = which(X$selection.id == 
                                                                                   spc.cmbs[j, 2]), path = path)
                                 srt<-min(clp1@samp.rate,clp2@samp.rate)
                                 if(clp1@samp.rate!=srt)clp1<-seewave::resamp(clp1,g=srt,output="Wave")
                                 if(clp2@samp.rate!=srt)clp2<-seewave::resamp(clp2,g=srt,output="Wave")
                                 spc1 <- spc_FUN(clp1, wlg = wl, 
                                                 ovl = ovlp, w = wn, nbnds = nbands)
                                 spc2 <- spc_FUN(clp2, wlg = wl, 
                                                 ovl = ovlp, w = wn, nbnds = nbands)
                               }
                               XC_FUN(spc1 = spc1, spc2 = spc2, b = BP, cm = cor.meth,step=step)
                             })
  xcrrs <- xcrrs[order(ord.shuf)]
  mx.xcrrs <- sapply(xcrrs, max, na.rm = TRUE)
  if (is.null(compare.matrix)) {
    mat <- matrix(nrow = nrow(X), ncol = nrow(X))
    mat[] <- 1
    colnames(mat) <- rownames(mat) <- X$selection.id
    mat[lower.tri(mat, diag = FALSE)] <- mx.xcrrs
    mat <- t(mat)
    mat[lower.tri(mat, diag = FALSE)] <- mx.xcrrs
    if (na.rm) {
      com.case <- intersect(rownames(mat)[stats::complete.cases(mat)], 
                            colnames(mat)[stats::complete.cases(t(mat))])
      if (length(which(is.na(mat))) > 0) 
        warning(paste(length(which(is.na(mat))), "pairwise comparisons failed and were removed"))
      mat <- mat[rownames(mat) %in% com.case, colnames(mat) %in% 
                   com.case]
      if (nrow(mat) == 0) 
        stop("Not selections remained after removing NAs (na.rm = TRUE)")
    }
  }
  else {
    mat <- data.frame(compare.matrix, score = mx.xcrrs)
    if (na.rm) 
      mat <- mat[!is.infinite(mat$scores), ]
  }
  if (output == "list") {
    cor.lst <- lapply(1:nrow(spc.cmbs.org), function(x) {
      durs <- c((X$end - X$start)[X$selection.id == spc.cmbs.org[x, 
                                                                 1]], (X$end - X$start)[X$selection.id == spc.cmbs.org[x, 
                                                                                                                       2]])
      df <- data.frame(dyad = paste(spc.cmbs.org[x, ], 
              collapse = "/"), sound.files = spc.cmbs.org[x, 
              which.max(durs)], template = spc.cmbs.org[x, 
              which.min(durs)], time = if (!is.null(xcrrs[[x]])) 
              c(X$start[X$selection.id == spc.cmbs.org[x,1]], X$start[X$selection.id == spc.cmbs.org[x, 
             2]])[which.max(durs)] + seq(min(durs)/2, max(durs) - 
             min(durs)/2, length.out = length(xcrrs[[x]]))
                       else NA, score = if (!is.null(xcrrs[[x]])) 
                         xcrrs[[x]]
                       else NA)
      return(df)
    })
    cor.table <- do.call(rbind, cor.lst)
    if (na.rm) {
      if (exists("com.case")) 
        cor.table <- cor.table[cor.table$sound.files %in% 
                                 com.case & cor.table$template %in% com.case, 
        ]
      errors <- cor.table[is.na(cor.table$score), ]
      errors$score <- NULL
      cor.table <- cor.table[!is.infinite(cor.table$score), 
      ]
    }
  }
  if (output == "cor.mat") 
    return(mat)
  else {
    output_list <- list(max.xcorr.matrix = mat, scores = cor.table, 
                        selection.table = X, hop.size.ms = read_wave(X, 
                                                                     1, header = TRUE, path = path)$sample.rate/wl, 
                        errors = if (na.rm) errors else NA)
    class(output_list) <- c("list", "xcorr.output")
    return(output_list)
  }
}


write.kaleidoscope<-function(X,path=NULL,file=NULL){
  if(is.null(file))
     stop("no output file name provided")
  if (is.null(path)) 
    path <- getwd()
  else if (!dir.exists(path)) 
    stop("'path' provided does not exist")
  else path <- normalizePath(path)  
  if (!any(is.data.frame(X), is_selection_table(X), is_extended_selection_table(X))) 
    stop("X is not of a class 'data.frame', 'selection_table' or 'extended_selection_table'")
  if (!all(c("sound.files", "selec", "start", "end") %in% 
           colnames(X))) 
    stop(paste(paste(c("sound.files", "selec", "start", 
                       "end")[!(c("sound.files", "selec", "start", "end") %in% 
                                  colnames(X))], collapse = ", "), "column(s) not found in data frame"))
  if (any(is.na(c(X$end, X$start)))) 
    stop("NAs found in start and/or end")
  if (any(!is(X$end, "numeric"), !is(X$start, "numeric"))) 
    stop("'start' and 'end' must be numeric")
  if (any(X$end - X$start <= 0)) 
    stop(paste("The start is higher than or equal to the end in", 
               length(which(X$end - X$start <= 0)), "case(s)"))   
  
  #export Kaleidoscope
  kdf<-data.frame(FOLDER=path,"IN FILE"=X$sound.files,CHANNEL=0,OFFSET=X$start,DURATION=X$end-X$start,"MANUAL ID"=X$score)
  kdf$FOLDER<-gsub("/","\\\\",kdf$FOLDER)
  if(dirname(file)=="." & !is.null(path))
    file=paste(path,file,sep="/")
  write.table(kdf,file=file,sep=",",row.names = F,quote = F,col.names=c("FOLDER","IN FILE","CHANNEL","OFFSET","DURATION","MANUAL ID"))
}