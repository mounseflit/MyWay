/*const possibleEmojis = [
  '🐀','🐁','🐭','🐹','🐂','🐃','🐄','🐮','🐅','🐆','🐯','🐇','🐐','🐑','🐏','🐴',
  '🐎','🐱','🐈','🐰','🐓','🐔','🐤','🐣','🐥','🐦','🐧','🐘','🐩','🐕','🐷','🐖',
  '🐗','🐫','🐪','🐶','🐺','🐻','🐨','🐼','🐵','🙈','🙉','🙊','🐒','🐉','🐲','🐊',
  '🐍','🐢','🐸','🐋','🐳','🐬','🐙','🐟','🐠','🐡','🐚','🐌','🐛','🐜','🐝','🐞',
];
function randomEmoji() {
  var randomIndex = Math.floor(Math.random() * possibleEmojis.length);
  return possibleEmojis[randomIndex];
}

const emoji = randomEmoji();*/

const emoji = "👤";
const name = prompt("What's your name?");

if (!location.hash) {
  location.hash = Math.floor(Math.random() * 0xffffff).toString(16);
}
const chatHash = location.hash.substring(1);

const drone = new ScaleDrone("dLoPljYYeFH4tKGn");

const roomName = "observable-" + chatHash;

let room;

const configuration = {
  iceServers: [
    {
      url: "stun:stun.l.google.com:19302",
    },
  ],
};

let pc;

let dataChannel;

drone.on("open", (error) => {
  if (error) {
    return console.error(error);
  }
  room = drone.subscribe(roomName);
  room.on("open", (error) => {
    if (error) {
      return console.error(error);
    }
    console.log("Connected to signaling server");
  });

  room.on("members", (members) => {
    if (members.length >= 3) {
      return alert("The room is full");
    }

    const isOfferer = members.length === 2;
    startWebRTC(isOfferer);
  });
});

function sendSignalingMessage(message) {
  drone.publish({
    room: roomName,
    message,
  });
}

function startWebRTC(isOfferer) {
  console.log("Starting in as", isOfferer ? "offerer" : "waiter");
  pc = new RTCPeerConnection(configuration);

  pc.onicecandidate = (event) => {
    if (event.candidate) {
      sendSignalingMessage({ candidate: event.candidate });
    }
  };

  if (isOfferer) {
    pc.onnegotiationneeded = () => {
      pc.createOffer(localDescCreated, (error) => console.error(error));
    };
    dataChannel = pc.createDataChannel("chat");
    setupDataChannel();
  } else {
    pc.ondatachannel = (event) => {
      dataChannel = event.channel;
      setupDataChannel();
    };
  }

  startListentingToSignals();
}

function startListentingToSignals() {
  room.on("data", (message, client) => {
    if (client.id === drone.clientId) {
      return;
    }
    if (message.sdp) {
      pc.setRemoteDescription(
        new RTCSessionDescription(message.sdp),
        () => {
          console.log("pc.remoteDescription.type", pc.remoteDescription.type);
          if (pc.remoteDescription.type === "offer") {
            console.log("Answering offer");
            pc.createAnswer(localDescCreated, (error) => console.error(error));
          }
        },
        (error) => console.error(error)
      );
    } else if (message.candidate) {
      pc.addIceCandidate(new RTCIceCandidate(message.candidate));
    }
  });
}

function localDescCreated(desc) {
  pc.setLocalDescription(
    desc,
    () => sendSignalingMessage({ sdp: pc.localDescription }),
    (error) => console.error(error)
  );
}

function setupDataChannel() {
  checkDataChannelState();
  dataChannel.onopen = checkDataChannelState;
  dataChannel.onclose = checkDataChannelState;
  dataChannel.onmessage = (event) =>
    insertMessageToDOM(JSON.parse(event.data), false);
}

function checkDataChannelState() {
  console.log("the channel state is:", dataChannel.readyState);
  if (dataChannel.readyState === "open") {
    insertMessageToDOM({ content: "the channel is now open" });
  }
}

function insertMessageToDOM(options, isFromMe) {
  const template = document.querySelector('template[data-template="message"]');
  const nameEl = template.content.querySelector(".message__name");

    if (isFromMe) {
      nameEl.innerText = "👤" + " " + options.name;
    } else {
      nameEl.innerText = "🗣️" + " " + options.name;
      if (!options.name) {
        nameEl.innerText = "🤖" + " " + "Bot";}
    }
  template.content.querySelector(".message__bubble").innerText =
    options.content;
  const clone = document.importNode(template.content, true);
  const messageEl = clone.querySelector(".message");
  if (isFromMe) {
    messageEl.classList.add("message--mine");
  } else {
    messageEl.classList.add("message--theirs");
  }
  const messagesEl = document.querySelector(".messages");
  messagesEl.appendChild(clone);
  messagesEl.scrollTop = messagesEl.scrollHeight - messagesEl.clientHeight;
}

/*function insertMessageToDOM(options, isFromMe) {
  const template = document.querySelector('template[data-template="message"]');
  const nameEl = template.content.querySelector('.message__name');
  if (options.emoji || options.name) {
    nameEl.innerText = options.emoji + ' ' + options.name;
  }
  template.content.querySelector('.message__bubble').innerText = options.content;
  const clone = document.importNode(template.content, true);
  const messageEl = clone.querySelector('.message');
  if (isFromMe) {
    messageEl.classList.add('message--mine');
  } else {
    messageEl.classList.add('message--theirs');
  }

  const messagesEl = document.querySelector('.messages');
  messagesEl.appendChild(clone);

  messagesEl.scrollTop = messagesEl.scrollHeight - messagesEl.clientHeight;
}*/


const form = document.querySelector("form");
form.addEventListener("submit", () => {
  const input = document.querySelector('input[type="text"]');
  const value = input.value;
  input.value = "";

  const data = {
    name,
    content: value,
  };

  dataChannel.send(JSON.stringify(data));

  insertMessageToDOM(data, true);
});

insertMessageToDOM({ content: "Chat URL is " + location.href });
