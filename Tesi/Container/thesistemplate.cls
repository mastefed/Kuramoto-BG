% INJ
% LaTeX class for BSc/MSc theses at dfa.unict.it
% (c) 2018, 2019, 2020, 2021 G. G. N. Angilella <giuseppe.angilella@ct.infn.it>
\NeedsTeXFormat{LaTeX2e}
%\ProvidesClass{dfaunictthesis}[2018/10/14 LaTeX class for BSc/MSc theses at dfa.unict.it]
%\ProvidesClass{dfaunictthesis}[2019/03/10 LaTeX class for BSc/MSc theses at dfa.unict.it]
\ProvidesClass{Container/thesistemplate}[2021/03/25 LaTeX class for BSc/MSc theses at dfa.unict.it]
\LoadClass[12pt,twoside]{book}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}
\RequirePackage[a-1b]{pdfx}
\RequirePackage{graphicx}
\RequirePackage{amsfonts}
\RequirePackage{amsbsy}
\RequirePackage{environ}
\RequirePackage[sc]{mathpazo}
\linespread{1.05}         % Palladio needs more leading (space between lines)


%\usepackage{lipsum}
%\usepackage{verbatim}

\RequirePackage{xifthen}
\RequirePackage{refcount}
\RequirePackage{xstring}

\newcommand{\thesistype}{BSc}

\newcommand\aayear[1]{\renewcommand\@aayear{#1}}
\newcommand\@aayear{\@latex@error{No \noexpand\aayear given}\@ehc}

\newcommand\phdname[1]{\renewcommand\@phdname{#1}}
\newcommand\@phdname{\@latex@error{No \noexpand\phdname given}\@ehc}

\DeclareOption{BSc}{\renewcommand{\thesistype}{BSc}}
\DeclareOption{MSc}{\renewcommand{\thesistype}{MSc}}
\DeclareOption{VO}{\renewcommand{\thesistype}{VO}}
\DeclareOption{PhD}{\renewcommand{\thesistype}{PhD}}
%\DeclareOption{italian}{\PassOptionsToPackage{italian}{babel}\selectlanguage{italian}}
%\DeclareOption{english}{\PassOptionsToPackage{english}{babel}\selectlanguage{english}}
\DeclareOption{italian}{\AtBeginDocument{\selectlanguage{italian}}}
\DeclareOption{english}{\AtBeginDocument{\selectlanguage{english}}}
%\DeclareOption{red}{\renewcommand{\headlinecolor}{\color{slcolor}}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax


\providecommand*{\supervisorname}{Supervisors}
\providecommand*{\cosupervisorname}{Co-Supervisors}
\providecommand*{\CdLname}{BSc programme in Physics}
\providecommand*{\thesistypename}{final project}
\providecommand*{\aaname}{academic year}

\phdname{\iflanguage{italian}{fisica}{physics}}

\newcounter{supervisorsno}
\newcounter{cosupervisorsno}

\newcommand\@supervisors{\@latex@error{No supervisors given}\@ehc}
\NewEnviron{supervisors}{\global\let\@supervisors\BODY}

\newcommand\@cosupervisors{}
\newcommand\@cosupervisorscheck{NOCOSUPERVISORS}
\NewEnviron{cosupervisors}{\global\let\@cosupervisors\BODY\global\let\@cosupervisorscheck\@empty}

\newcommand{\supervisor}[3]{\refstepcounter{supervisorsno}#1 \> #2 #3 \\}

\newcommand{\cosupervisor}[3]{\refstepcounter{cosupervisorsno}#1 \> #2 #3 \\}

\newcommand{\svnon}{}

\newcommand{\cosvnon}{}

\AtBeginDocument{%
    \StrBehind{\getrefbykeydefault{svno}{anchor}{}}{.}[\svnon]
    \StrBehind{\getrefbykeydefault{cosvno}{anchor}{}}{.}[\cosvnon]
    \iflanguage{italian}{%
        \ifthenelse{\equal{\svnon}{1}}{%
            \renewcommand*\supervisorname{Relatore}%
        }{%
            \renewcommand*\supervisorname{Relatori}%
        }
        \ifthenelse{\equal{\cosvnon}{1}}{%
            \renewcommand*\cosupervisorname{Correlatore}%
        }{%
            \renewcommand*\cosupervisorname{Correlatori}%
        }
        \renewcommand*\aaname{anno accademico}%
        \ifthenelse{\equal{\thesistype}{BSc}}{%
            \renewcommand{\CdLname}{corso di laurea in fisica}%
            \renewcommand{\thesistypename}{elaborato finale}%
        }{\relax}
        \ifthenelse{\equal{\thesistype}{MSc}}{%
            \renewcommand{\CdLname}{corso di laurea magistrale in fisica}%
            \renewcommand{\thesistypename}{elaborato finale}%
        }{\relax}
        \ifthenelse{\equal{\thesistype}{VO}}{%
            \renewcommand{\CdLname}{corso di laurea in fisica}%
            \renewcommand{\thesistypename}{tesi di laurea}%
        }{\relax}
        \ifthenelse{\equal{\thesistype}{PhD}}{%
            \renewcommand{\CdLname}{dottorato di ricerca in \@phdname}%
            \renewcommand{\thesistypename}{tesi di dottorato}%
        }{\relax}
    }{\relax}%
    \iflanguage{english}{%
        \ifthenelse{\equal{\svnon}{1}}{%
            \renewcommand*\supervisorname{Supervisor}%
        }{%
            \renewcommand*\supervisorname{Supervisors}%
        }
        \ifthenelse{\equal{\cosvnon}{1}}{%
            \renewcommand*\cosupervisorname{Co-Supervisor}%
        }{%
            \renewcommand*\cosupervisorname{Co-Supervisors}%
        }
        \renewcommand*\aaname{academic year}%
        \ifthenelse{\equal{\thesistype}{BSc}}{%
            \renewcommand{\CdLname}{BSc programme in physics}%
            \renewcommand{\thesistypename}{final project}%
        }{\relax}
        \ifthenelse{\equal{\thesistype}{MSc}}{%
            \renewcommand{\CdLname}{MSc programme in physics}%
            \renewcommand{\thesistypename}{final project}%
        }{\relax}
        \ifthenelse{\equal{\thesistype}{VO}}{%
            \renewcommand{\CdLname}{degree course in physics}%
            \renewcommand{\thesistypename}{degree thesis}%
        }{\relax}
        \ifthenelse{\equal{\thesistype}{PhD}}{%
            \renewcommand{\CdLname}{PhD programme in \@phdname}%
            \renewcommand{\thesistypename}{PhD thesis}%
        }{\relax}
    }{\relax}%
}

\newcommand{\maketitlepage}{%
    \begin{titlepage}
        \begin{center}
%            \includegraphics[width=16em]{unict_dfa}\\
            \begin{minipage}[c]{0.45\textwidth}
               \begin{center}
                  \includegraphics[width=0.8\textwidth]{Images/logounipi}
               \end{center}
            \end{minipage}
            \\
            \medskip
            {\sc\CdLname}\\
            \hbox to \textwidth{\hrulefill}
            \vspace{3truecm}
            {\sc\@author}
            \vfill
            \uppercase{\sc\@title}
            \vfill
            \centerline{\hbox to 3.5truecm{\hrulefill}}
            \medskip
            {\sc\thesistypename}\\
            \centerline{\hbox to 3.5truecm{\hrulefill}}
            \vfill
            \begin{flushright}
                \begin{minipage}{0.4\textwidth}
                    \begin{tabbing}
                        Chiar.mo \= Prof. P. Pallino \kill
                        \supervisorname: \> \\
                        \@supervisors
                        \label{svno}%
                        \ifthenelse{\equal{\expandafter\@cosupervisorscheck}{NOCOSUPERVISORS}}{%
                           \relax
                        }{%
                           \medskip\\
                           \cosupervisorname: \> \\
                           \@cosupervisors
                           \label{cosvno}
                        }
                    \end{tabbing}
                \end{minipage}
            \end{flushright}
            \vfill
            \hbox to \textwidth{\hrulefill}
            {\sc\aaname{} \@aayear}
        \end{center}
    \end{titlepage}
}