(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const el = (tag, cls, value) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (value !== undefined) node.textContent = value;
    return node;
  };
  let workspace, selected = null, detail = null, panel = 'match', expanded = false;
  let searchVersion = 0, detailVersion = 0, busy = false;
  const drafts = new Map();
  const degrees = {1:'Associate degree', 2:'Bachelor’s degree', 3:'Master’s degree', 4:'Doctorate'};

  function error(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
  async function api(path, data) {
    const response = await fetch(path, data === undefined ? {} : {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The request could not be completed.');
    return result;
  }
  async function act(action) {
    if (busy) return;
    busy = true; error(''); $('workspace').setAttribute('aria-busy','true');
    const controls = [...document.querySelectorAll('button,input,select,textarea')];
    const previous = controls.map(n => n.disabled);
    controls.forEach(n => { n.disabled = true; });
    try { await action(); } catch (err) { error(err.message); }
    finally {
      busy = false; controls.forEach((n,i) => { n.disabled = previous[i]; });
      $('workspace').setAttribute('aria-busy','false');
      configureChat();
    }
  }
  function configureChat() {
    const enabled = workspace?.chat_enabled && !busy;
    $('chat-message').disabled = !enabled; $('chat-send').disabled = !enabled;
    $('chat-mode').textContent = workspace?.chat_enabled
      ? 'Voyager connected. Answers are checked against retrieved profile and posting facts.'
      : 'Chat is off in this run. Search, explanations, gaps, and template drafts work without a model.';
  }
  function chips(parent, values, kind='') {
    const row = el('div','chips');
    values.forEach(value => row.append(el('span','chip '+kind,value)));
    parent.append(row);
  }
  function skillBlock(parent, title, values, kind, empty) {
    const block = el('section','detail-block');
    const heading = el('h3','block-label',title);
    heading.append(el('span','count-badge',values.length)); block.append(heading);
    if (values.length) chips(block,values,kind);
    else block.append(el('p','empty-skills',empty));
    parent.append(block);
  }
  function renderProfile() {
    const profile = workspace.profile;
    $('resume-name').textContent = profile.cv_id;
    $('profile-count').textContent = `${profile.skills.length} extracted skills`;
    const processing = workspace.profile_processing;
    $('profile-processing').hidden = !processing;
    $('profile-processing').textContent = processing
      ? processing.warnings.length ? processing.warnings.join('. ') : 'Resume loaded. Review the extracted facts before using your matches.'
      : '';
    $('profile-picker').replaceChildren(...workspace.profiles.map(p => {
      const option = el('option','',p.cv_id); option.value = p.index; return option;
    }));
    $('profile-picker').value = workspace.profile_index;
    $('profile-skills').replaceChildren();
    (expanded ? profile.skills : profile.skills.slice(0,7)).forEach(s => $('profile-skills').append(el('span','chip',s.label)));
    $('more-skills').textContent = expanded ? 'Show fewer skills' : `Show all ${profile.skills.length} skills`;
    $('more-skills').hidden = profile.skills.length <= 7;
    $('profile-years').textContent = profile.years_experience === null ? 'Years of experience: not documented' : `${profile.years_experience} years documented`;
    $('profile-degree').textContent = profile.degree_rank === null ? 'Degree: not documented' : degrees[profile.degree_rank] || 'Degree documented';
    $('catalog-name').textContent = workspace.catalog_name;
    $('catalog-count').textContent = `${workspace.catalog_size} postings available`;
    $('system').replaceChildren(...workspace.systems.map(s => {
      const option = el('option','',`${s.id} · ${s.name}`); option.value = s.id; return option;
    }));
    $('system').value = workspace.system;
    $('mode').textContent = workspace.profile_demo && workspace.jobs_demo ? 'Demo · synthetic job data'
      : workspace.profile_demo || workspace.jobs_demo ? 'Mixed sources · includes demo data' : 'Your data workspace';
    $('footer-status').textContent = `${workspace.system} ranking · ${workspace.graph_mode}`;
    configureChat();
  }
  async function refresh(value) {
    workspace = value || await api('/api/state');
    searchVersion++; detailVersion++; selected = null; detail = null; panel = 'match';
    drafts.clear(); $('chat-history').replaceChildren(); $('chat-message').value = '';
    $('selected-detail').hidden = true; $('empty-detail').hidden = false;
    renderProfile(); await search();
  }
  async function search() {
    const version = ++searchVersion;
    const data = await api('/api/search',{query:$('search').value,top_k:20});
    if (version !== searchVersion) return;
    const list = $('job-list'); list.replaceChildren();
    $('result-count').textContent = `${data.results.length} recommendations · ${workspace.catalog_size} postings in catalog`;
    for (const job of data.results) {
      const button = el('button','job-card'); button.type = 'button'; button.dataset.jobId = job.job_id;
      button.setAttribute('aria-pressed',String(job.job_id === selected));
      button.setAttribute('aria-label',`${job.title} at ${job.company || 'Company not listed'}`);
      button.append(el('p','job-rank',`Result #${job.rank}`), el('p','job-title',job.title),
        el('p','job-company',[job.company,job.location].filter(Boolean).join(' · ')),
        el('p','job-skill-count',`${job.matched_required_count} of ${job.required_count} required skills documented`),
        el('p','job-gap-count',job.missing_required.length ? `${job.missing_required.length} required skill${job.missing_required.length===1?'':'s'} to review` : 'Review the full requirements'));
      button.addEventListener('click',() => act(() => selectJob(job.job_id))); list.append(button);
    }
    if (!data.results.length) {
      list.append(el('p','muted',data.note || 'No matching postings. Try fewer keywords.'));
      selected = null; detail = null; detailVersion++;
      $('selected-detail').hidden = true; $('empty-detail').hidden = false;
    } else if (!data.results.some(job => job.job_id === selected)) {
      await selectJob(data.results[0].job_id);
    }
  }
  async function selectJob(id) {
    const version = ++detailVersion;
    const data = await api('/api/detail?job_id='+encodeURIComponent(id));
    if (version !== detailVersion) return;
    selected = id; detail = data; panel = 'match';
    document.querySelectorAll('.job-card').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.jobId===id)));
    $('selected-detail').hidden = false; $('empty-detail').hidden = true;
    const match = detail.explanation;
    $('selected-rank').textContent = match.rank === null ? 'Outside the current retrieval depth' : `Overall recommendation #${match.rank}`;
    $('selected-title').textContent = match.title;
    $('selected-company').textContent = [match.company,detail.location].filter(Boolean).join(' · ');
    const link = $('posting-link'); link.hidden = true; link.removeAttribute('href');
    try { const url = new URL(match.url); if (['https:','http:'].includes(url.protocol)) { link.href=url.href; link.hidden=false; } } catch (_) {}
    renderDetail();
  }
  function renderDetail() {
    if (!detail) return;
    const match = detail.explanation, content = $('detail-content'); content.replaceChildren();
    document.querySelectorAll('[data-panel]').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.panel===panel)));
    if (panel === 'match') {
      skillBlock(content,'Required skills documented',detail.matched_required,'good','No required skills documented in this extracted profile.');
      skillBlock(content,'Required · not documented',match.missing_required,'gap','No required skill gaps found in this extracted profile.');
      if (detail.matched_preferred.length) skillBlock(content,'Preferred skills documented',detail.matched_preferred,'good','');
      if (match.missing_preferred.length) skillBlock(content,'Preferred · not documented',match.missing_preferred,'gap','');
      if (match.near_miss_skills.length) {
        skillBlock(content,'Related skills to review',match.near_miss_skills.map(s=>`${s.your_skill} → ${s.job_skill}`),'','');
      }
      const section = el('section','detail-block'); section.append(el('h3','','Other requirements'));
      for (const key of ['years_experience','degree_level']) {
        const req = match.requirements[key]; const row = el('div','requirement');
        const name = key === 'years_experience' ? 'Experience' : 'Degree';
        const required = req.required === null ? `${name}: employer minimum not stated`
          : key === 'years_experience' ? `${req.required}+ years of experience` : degrees[req.required] || 'Degree requirement';
        const label = {'not stated':'No minimum provided','unknown from resume':'Needs review · not documented in resume',
          meets:'Documented in resume',below:'Below the stated minimum'}[req.status];
        row.append(el('span','',required),el('span',['unknown from resume','below'].includes(req.status)?'needs-review':'',label)); section.append(row);
      }
      section.append(el('p','small muted','“Not documented” means the parsed profile does not contain the fact. Check your resume before drawing a conclusion.'));
      content.append(section);
      const diagnostics = el('details'); diagnostics.append(el('summary','','Ranking details'));
      diagnostics.append(el('p','small muted',`${match.system} score: ${match.score} · Graph skill score: ${match.kg_score} (0–1). These are ranking signals, not a probability of getting the job.`));
      Object.entries(match.rankers).forEach(([name, signal]) => diagnostics.append(
        el('p','small muted',`${name}: rank ${signal.rank ?? 'outside retrieval depth'} · score ${signal.score ?? 'not retrieved'}`)));
      content.append(diagnostics);
    } else if (panel === 'evidence') {
      const posting = el('section','detail-block'); posting.append(el('h3','','Original posting text'));
      posting.append(el('p','small muted',detail.source || 'Source label not provided'));
      posting.append(el('div','source-text',detail.description || 'Source text was not included in this export. Load it with the source postings CSV to inspect the original wording.'));
      content.append(posting);
      if(match.best_matching_text?.text){
        const segment=el('details');segment.append(el('summary','','Top contributing posting segment'));
        segment.append(el('div','source-text',match.best_matching_text.text));content.append(segment);
      }
      const quotes = el('details'); quotes.append(el('summary','','Resume source excerpts'));
      if (detail.resume_evidence.length) detail.resume_evidence.forEach(q=>quotes.append(el('p','source-text',q)));
      else quotes.append(el('p','small muted','This parsed resume does not include source excerpts.'));
      content.append(quotes);
      const facts = el('details'); facts.append(el('summary','','Normalized facts and graph paths'));
      facts.append(el('p','small muted','These facts come from the loaded ontology and matching module; they are separate from source quotations.'));
      match.context.forEach(f=>{const p=el('p','evidence-fact');p.append(el('span','fact-id',f.id),document.createTextNode(f.text));facts.append(p);});
      content.append(facts);
    } else { renderDraft(content); }
  }
  function renderDraft(content) {
    content.append(el('p','review-label','Draft only · Review and edit before use'));
    let draft = drafts.get(selected);
    const label=el('label','field-label','Recipient');label.htmlFor='recipient';
    const recipient=el('select');recipient.id='recipient';
    for(const name of ['recruiter','referral']){const o=el('option','',name==='recruiter'?'Recruiter':'Referral contact');o.value=name;recipient.append(o);}
    recipient.value=draft?.recipient || 'recruiter';
    const generate=el('button','primary',draft?'Regenerate draft':'Create draft');generate.type='button';
    const actions=el('div','draft-actions');actions.append(generate);content.append(label,recipient,actions);
    generate.addEventListener('click',()=>act(async()=>{
      const id=selected, result=await api('/api/draft',{job_id:id,recipient:recipient.value});
      drafts.set(id,result);if(selected===id && panel==='draft')renderDetail();
    }));
    if(!draft){content.append(el('p','small muted','Create an introduction from the documented profile and posting facts.'));return;}
    const subjectLabel=el('label','field-label','Subject');subjectLabel.htmlFor='draft-subject';
    const subject=el('input');subject.id='draft-subject';subject.value=draft.subject;
    const messageLabel=el('label','field-label','Message');messageLabel.htmlFor='draft-message';
    const message=el('textarea','draft-text');message.id='draft-message';message.value=draft.message;
    subject.addEventListener('input',()=>{draft.subject=subject.value;}); message.addEventListener('input',()=>{draft.message=message.value;});
    const status=el('p','small muted',`${draft.source==='template'?'Template from profile facts':'Voyager draft'} · ${draft.verification.ok?'Skill/years check passed':'Needs review'} · Edits need your review.`);
    const copy=el('button','primary','Copy draft');copy.type='button';
    const copyStatus=el('p','small muted','Nothing is sent automatically.');copyStatus.setAttribute('role','status');
    copy.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(subject.value+'\n\n'+message.value);copyStatus.textContent='Copied. Review before sending.';}catch(_){message.focus();message.select();copyStatus.textContent='Select and copy the subject and message with your keyboard.';}});
    const copyActions=el('div','draft-actions');copyActions.append(copy,copyStatus);
    content.append(subjectLabel,subject,messageLabel,message,status,copyActions);
  }
  $('more-skills').addEventListener('click',()=>{expanded=!expanded;renderProfile();});
  $('search-form').addEventListener('submit',e=>{e.preventDefault();act(search);});
  $('profile-picker').addEventListener('change',()=>act(async()=>refresh(await api('/api/profile',{index:Number($('profile-picker').value)}))));
  $('system').addEventListener('change',()=>act(async()=>refresh(await api('/api/system',{system:$('system').value}))));
  $('reset').addEventListener('click',()=>act(async()=>{$('search').value='';await refresh(await api('/api/reset',{}));}));
  document.querySelectorAll('[data-panel]').forEach(button=>button.addEventListener('click',()=>{panel=button.dataset.panel;renderDetail();}));
  for(const [id,route] of [['resume-upload','/api/profile'],['jobs-upload','/api/jobs']]){
    $(id).addEventListener('change',()=>act(async()=>{
      const file=$(id).files[0];if(!file)return;
      try{if(file.size>6*1024*1024)throw new Error('Choose a file under 6 MB.');
        await refresh(await api(route,{content:await file.text()}));}
      finally{$(id).value='';}
    }));
  }
  $('chat-form').addEventListener('submit',e=>{e.preventDefault();act(async()=>{
    const message=$('chat-message').value.trim();if(!message)return;
    const reply=await api('/api/chat',{message});
    $('chat-history').append(el('p','chat-entry user',message),el('p','chat-entry',reply.answer));
    if(!reply.verified || reply.warnings.length)$('chat-history').append(el('p','warning',reply.warnings.join(' · ') || 'This answer needs review.'));
    $('chat-message').value='';
  });});
  act(()=>refresh());
})();
